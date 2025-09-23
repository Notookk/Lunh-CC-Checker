import os
import re
import json
import random
import string
import logging
import tempfile
import datetime
import asyncio
from typing import List, Dict, Optional

import fitz
import docx
import aiofiles
import motor.motor_asyncio

from telegram import Update, InputFile
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------------- CONFIG ----------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "cc_checker_bot")
OWNER_IDS = [111111111, 222222222]
CHANNEL_ID = -1001234567890
LIVE_PROBABILITY = 0.20

#--------Limits / Safety--------
MAX_FILE_SIZE_MB = 8
RATE_LIMIT_SECONDS = 60
STORE_FULL_RAW_LINES = False  # NEVER store raw full card lines if False (we store masked only)
MASK_REVEAL_LAST = 4  # how many digits to reveal in mask (default last 4)

# ---------------- LOGGING ----------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------- MONGO ----------------
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DB_NAME]
sudo_collection = db["sudo_users"]
checks_collection = db["checks"]

# ---------------- IN-MEMORY RATE LIMITING ----------------
_last_bulk_time: Dict[int, float] = {}  # user_id -> timestamp


# ---------------- CARD VALIDATOR CLASS ----------------
class CardValidator:
    def __init__(self, live_probability: float = LIVE_PROBABILITY):
        self.live_probability = live_probability
        # Updated brand patterns (non-exhaustive but closer to real ranges)
        # Visa: 13, 16, or 19 digits starting with 4
        visa_re = r"^4\d{12}(?:\d{3}){0,2}$"
        # Mastercard: 51-55 or 2221-2720 (16 digits)
        mc_re = (
            r"^(?:5[1-5]\d{14}|"
            r"2(?:2(?:2[1-9]|[3-9]\d)|[3-6]\d{2}|7(?:[01]\d|20))\d{12})$"
        )
        # AmEx: 34 or 37, 15 digits
        amex_re = r"^3[47]\d{13}$"
        # Discover: 6011, 65, 644-649, 622126-622925 (16 digits typical)
        discover_re = (
            r"^(?:6011\d{12}|65\d{14}|64[4-9]\d{13}|"
            r"622(?:12[6-9]|1[3-9]\d|[2-8]\d{2}|9(?:[01]\d|2[0-5]))\d{10})$"
        )
        # Diners Club: 300-305, 3095, 36, 38-39 (14 digits)
        diners_re = r"^(?:30[0-5]\d{11}|3095\d{10}|36\d{12}|3[89]\d{12})$"
        # JCB: 3528-3589 (16 digits)
        jcb_re = r"^35(?:2[89]|[3-8]\d)\d{12}$"
        self.card_patterns = {
            "visa": re.compile(visa_re),
            "mastercard": re.compile(mc_re),
            "amex": re.compile(amex_re),
            "discover": re.compile(discover_re),
            "diners": re.compile(diners_re),
            "jcb": re.compile(jcb_re),
        }

    @staticmethod
    def _clean_number(num: str) -> str:
        return "".join(ch for ch in num if ch.isdigit())

    def detect_type(self, number: str) -> str:
        cleaned = self._clean_number(number)
        for name, pattern in self.card_patterns.items():
            if pattern.match(cleaned):
                return name
        return "unknown"

    def luhn_check(self, number: str) -> bool:
        cleaned = self._clean_number(number)
        if len(cleaned) < 13 or len(cleaned) > 19:
            return False
        total = 0
        double = False
        for ch in reversed(cleaned):
            d = int(ch)
            if double:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
            double = not double
        return total % 10 == 0

    def validate_expiry(self, month: str, year: str) -> bool:
        try:
            mm = int(month)
            yy = int(year)
            if len(year) == 2:
                yy += 2000
            # month validity
            if mm < 1 or mm > 12:
                return False
            now = datetime.datetime.utcnow()
            if yy < now.year or (yy == now.year and mm < now.month):
                return False
            return True
        except Exception:
            return False

    def validate_cvv(self, cvv: str, card_type: str) -> bool:
        if not cvv or not cvv.isdigit():
            return False
        if card_type == "amex":
            return len(cvv) == 4
        return len(cvv) == 3

    def simulate_status(self) -> str:
        return "LIVE" if random.random() < self.live_probability else "DEAD"

    @staticmethod
    def mask_number(num: str, reveal_last: int = MASK_REVEAL_LAST) -> str:
        cleaned = "".join(ch for ch in num if ch.isdigit())
        if not cleaned:
            return ""
        if len(cleaned) <= reveal_last:
            return "*" * len(cleaned)
        masked = "*" * (len(cleaned) - reveal_last) + cleaned[-reveal_last:]
        # group into 4-digit chunks for readability
        groups = [masked[max(i - 4, 0):i] for i in range(len(masked), 0, -4)][::-1]
        return " ".join(groups)

    def extract_parts(self, line: str) -> Dict[str, str]:
        """
        Extract parts from a line in a flexible way.
        Accepts separators like | , ; : / - space. Finds digit sequences and
        attempts to assign: number (13-19 digits), month (1-12), year (2 or 4 digits), cvv (3-4 digits).
        """
        # Gather all digit chunks with their positions to keep order
        matches = list(re.finditer(r"\d+", line))
        tokens = [m.group(0) for m in matches]

        # Fallback to simple split if nothing numeric found
        if not tokens:
            parts = [p.strip() for p in re.split(r"[|,;:/\s-]+", line) if p.strip()]
            return {
                "number": parts[0] if len(parts) > 0 else "",
                "month": parts[1] if len(parts) > 1 else "",
                "year": parts[2] if len(parts) > 2 else "",
                "cvv": parts[3] if len(parts) > 3 else "",
            }

        def pick_number() -> Optional[int]:
            # choose first token length 13..19
            for idx, t in enumerate(tokens):
                if 13 <= len(t) <= 19:
                    return idx
            # fallback: longest token
            return max(range(len(tokens)), key=lambda i: len(tokens)) if tokens else None

        n_idx = pick_number()
        number = tokens[n_idx] if n_idx is not None else ""

        # month after number
        m_idx = None
        for idx in range((n_idx or -1) + 1, len(tokens)):
            t = tokens[idx]
            if 1 <= len(t) <= 2:
                try:
                    mm = int(t)
                    if 1 <= mm <= 12:
                        m_idx = idx
                        break
                except Exception:
                    pass

        # year after month if possible
        y_idx = None
        search_start = (m_idx if m_idx is not None else (n_idx if n_idx is not None else -1)) + 1
        # prefer 4-digit year
        for idx in range(search_start, len(tokens)):
            t = tokens[idx]
            if len(t) == 4:
                y_idx = idx
                break
        if y_idx is None:
            for idx in range(search_start, len(tokens)):
                t = tokens[idx]
                if len(t) == 2:
                    y_idx = idx
                    break

        # cvv after year/month
        c_idx = None
        search_start = (y_idx if y_idx is not None else (m_idx if m_idx is not None else (n_idx if n_idx is not None else -1))) + 1
        for idx in range(search_start, len(tokens)):
            t = tokens[idx]
            if len(t) in (3, 4):
                c_idx = idx
                break

        return {
            "number": number or "",
            "month": tokens[m_idx] if m_idx is not None else "",
            "year": tokens[y_idx] if y_idx is not None else "",
            "cvv": tokens[c_idx] if c_idx is not None else "",
        }

    def validate_line(self, line: str) -> Dict:
        """
        Validate a single input line (format flexible but expects number|MM|YYYY|CVV).
        Returns a dict with masked number, type, valid, reason, and status.
        CVV will NOT be stored or returned in reports.
        """
        parts = self.extract_parts(line)
        raw_num = parts["number"]
        month = parts["month"]
        year = parts["year"]
        cvv = parts["cvv"]

        cleaned = self._clean_number(raw_num)
        masked = self.mask_number(cleaned)

        res = {
            "raw_line": line if STORE_FULL_RAW_LINES else None,  # only stored if explicitly allowed
            "number": cleaned,
            "masked": masked,
            "type": None,
            "valid": False,
            "reason": None,
            "status": None,
        }

        if not cleaned:
            res["reason"] = "Missing/invalid card number"
            return res

        res["type"] = self.detect_type(cleaned)

        if not self.luhn_check(cleaned):
            res["reason"] = "Failed Luhn check"
            return res

        if not self.validate_expiry(month, year):
            res["reason"] = "Invalid/expired expiry"
            return res

        if not self.validate_cvv(cvv, res["type"]):
            res["reason"] = "Invalid CVV"
            return res

        res["valid"] = True
        res["status"] = self.simulate_status()
        return res


validator = CardValidator()


# ---------------- MONGO / SUDO HELPERS ----------------
async def is_owner(user_id: int) -> bool:
    return user_id in OWNER_IDS


async def is_sudo(user_id: int) -> bool:
    # owners implicitly sudo
    if user_id in OWNER_IDS:
        return True
    doc = await sudo_collection.find_one({"user_id": user_id})
    return doc is not None


async def add_sudo_db(user_id: int, added_by: int):
    await sudo_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "user_id": user_id,
                "added_by": added_by,
                "added_at": datetime.datetime.utcnow(),
            }
        },
        upsert=True,
    )


async def remove_sudo_db(user_id: int):
    await sudo_collection.delete_one({"user_id": user_id})


async def list_sudo_db() -> List[Dict]:
    cursor = sudo_collection.find({}, {"_id": 0})
    return [doc async for doc in cursor]


# ---------------- FILE PARSING UTILITIES ----------------
async def parse_txt(path: str) -> List[str]:
    lines: List[str] = []
    try:
        async with aiofiles.open(path, mode="r", encoding="utf-8", errors="ignore") as f:
            async for ln in f:
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    except Exception:
        # fallback to sync read if aiofiles has issues
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f2:
                for ln in f2:
                    ln = ln.strip()
                    if ln:
                        lines.append(ln)
        except Exception:
            logger.exception("Failed to read txt file")
    return lines


async def parse_pdf(path: str) -> List[str]:
    lines: List[str] = []
    try:
        doc = fitz.open(path)
        for page in doc:
            text = page.get_text() or ""
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    except Exception as e:
        logger.exception("PDF parsing failed: %s", e)
    return lines


async def parse_docx(path: str) -> List[str]:
    lines: List[str] = []
    try:
        doc = docx.Document(path)
        for p in doc.paragraphs:
            ln = p.text.strip()
            if ln:
                lines.append(ln)
    except Exception as e:
        logger.exception("DOCX parsing failed: %s", e)
    return lines


# ---------------- HELPERS ----------------
def ext_from_filename(filename: str) -> str:
    parts = filename.rsplit(".", 1)
    return parts[-1].lower() if len(parts) > 1 else ""


def human_file_size_bytes(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


# ---------------- REPORT CREATION ----------------
async def create_masked_report_text(user_id: int, user_name: str, results: List[Dict]) -> str:
    now = datetime.datetime.utcnow().isoformat() + "Z"
    total = len(results)
    valid = sum(1 for r in results if r["valid"])
    live = sum(1 for r in results if r["valid"] and r["status"] == "LIVE")
    dead = sum(1 for r in results if r["valid"] and r["status"] == "DEAD")

    lines = [
        f"Card Check Report",
        f"Generated: {now}",
        f"User ID: {user_id}",
        f"User name: {user_name}",
        f"Total: {total}",
        f"Valid: {valid}",
        f"Live: {live}",
        f"Dead: {dead}",
        "-" * 40,
        "Results (masked) - CVVs omitted:",
    ]
    for r in results:
        if r["valid"]:
            lines.append(f"{r['masked']} | {r['type'].upper()} | {r['status']}")
        else:
            # show masked if available
            lines.append(f"{r['masked'] or 'N/A'} | INVALID | {r['reason']}")
    return "\n".join(lines)


# ---------------- RATE-LIMIT CHECK ----------------
def check_and_update_rate_limit(user_id: int) -> Optional[int]:
    """
    Return remaining seconds if rate limited, else None and update timestamp.
    """
    now = asyncio.get_event_loop().time()
    prev = _last_bulk_time.get(user_id)
    if prev:
        diff = now - prev
        if diff < RATE_LIMIT_SECONDS:
            return int(RATE_LIMIT_SECONDS - diff)
    _last_bulk_time[user_id] = now
    return None


# ---------------- BOT COMMANDS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to the Card Checker Bot.\n"
        "Use /help to see commands.\n\n"
        "⚠️ This bot processes text that may contain card data. CVVs are not preserved in reports by default."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - start the bot\n"
        "/help - this help\n"
        "/check <number|MM|YYYY|CVV> - check a single card (pipe separated). Anyone can use.\n"
        "Bulk check: upload a .txt, .pdf, or .docx file to the bot (sudo or owner only). The bot will parse lines (one per card) and produce a masked report.\n\n"
        "Owner commands (hardcoded OWNER_IDS): /addsudo <id>, /removesudo <id>\n"
        "Sudo commands: /listsudo\n"
        "Owner extra: /exportchecks <N> - export last N check metadata (owner-only)."
    )


async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /check <number|MM|YYYY|CVV>")
        return

    raw = " ".join(context.args).strip()
    res = validator.validate_line(raw)

    if res["valid"]:
        await update.message.reply_text(
            f"✅ VALID — {res['masked']} ({res['type'].upper()}) → {res['status']}"
        )
    else:
        await update.message.reply_text(
            f"❌ INVALID — {res['masked'] or res['number']} — Reason: {res['reason']}"
        )


# Bulk file handler - only sudo/owner allowed
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if msg is None or msg.document is None:
        return

    # Check sudo
    if not await is_sudo(user.id):
        await msg.reply_text("🚫 Bulk checking is allowed only for sudo/owner users.")
        return

    # Simple rate limit
    remaining = check_and_update_rate_limit(user.id)
    if remaining is not None:
        await msg.reply_text(
            f"⏳ You're doing checks too frequently. Wait {remaining} seconds and try again."
        )
        return

    doc = msg.document
    # check size
    if doc.file_size and (doc.file_size > MAX_FILE_SIZE_MB * 1024 * 1024):
        await msg.reply_text(
            f"❌ File too large ({human_file_size_bytes(doc.file_size)}). Limit is {MAX_FILE_SIZE_MB} MB."
        )
        return

    # Download file safely to temp
    try:
        await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)
    except Exception:
        pass
    try:
        tf = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tf.name
        tf.close()
        file_obj = await doc.get_file()
        await file_obj.download_to_drive(tmp_path)
    except Exception as e:
        logger.exception("Failed to download file: %s", e)
        await msg.reply_text("❌ Failed to download the file.")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return

    ext = ext_from_filename(doc.file_name or doc.file_unique_id)
    lines: List[str] = []
    try:
        if ext == "txt":
            lines = await parse_txt(tmp_path)
        elif ext in ("pdf", "xpdf"):
            lines = await parse_pdf(tmp_path)
        elif ext in ("docx", "doc"):
            lines = await parse_docx(tmp_path)
        else:
            # try text fallback
            lines = await parse_txt(tmp_path)
    except Exception as e:
        logger.exception("Error parsing uploaded file: %s", e)
        await msg.reply_text("❌ Error parsing file.")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return

    if not lines:
        await msg.reply_text("⚠️ No lines found in the uploaded file.")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return

    # Deduplicate by normalized number|MM|YYYY|CVV tokens
    seen_keys = set()
    uniq_lines: List[str] = []
    for ln in lines:
        parts = validator.extract_parts(ln)
        key = f"{validator._clean_number(parts.get('number',''))}|{parts.get('month','')}|{parts.get('year','')}|{parts.get('cvv','')}"
        if key not in seen_keys:
            seen_keys.add(key)
            uniq_lines.append(ln)

    # Process lines sequentially (can be parallelized safely but keep simple)
    results: List[Dict] = []
    for ln in uniq_lines:
        try:
            r = validator.validate_line(ln)
            # Ensure CVV isn't stored in result - validator doesn't store CVV
            results.append(r)
        except Exception as e:
            logger.exception("Validation error for line: %s", e)
            results.append({"masked": None, "valid": False, "reason": "Validation error"})

    # Create masked report text
    report_text = await create_masked_report_text(user.id, user.full_name or user.username or str(user.id), results)

    # Save to a temporary report file
    report_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    report_tmp_path = report_tmp.name
    report_tmp.write(report_text)
    report_tmp.close()

    # Reply to user with summary and report
    total = len(results)
    valid = sum(1 for r in results if r["valid"])
    live = sum(1 for r in results if r["valid"] and r["status"] == "LIVE")
    dead = sum(1 for r in results if r["valid"] and r["status"] == "DEAD")

    try:
        await msg.reply_text(
            f"✅ Bulk check completed.\nTotal: {total}\nValid: {valid}\nLive: {live}\nDead: {dead}\nSending masked report..."
        )
        await msg.reply_document(InputFile(report_tmp_path, filename=f"report_{doc.file_unique_id}.txt"))
    except Exception:
        logger.exception("Failed to send report to user.")

    # Send to channel (masked report)
    try:
        await context.bot.send_document(CHANNEL_ID, InputFile(report_tmp_path), caption=f"Masked report from {user.full_name or user.username or user.id}")
    except Exception:
        logger.exception("Failed to send report to channel.")

    # Save metadata to Mongo (do NOT store raw lines unless configured)
    try:
        metadata = {
            "user_id": user.id,
            "user_name": user.full_name or user.username or str(user.id),
            "timestamp": datetime.datetime.utcnow(),
            "file_name": doc.file_name,
            "total": total,
            "valid": valid,
            "live": live,
            "dead": dead,
            # store first 200 masked items as sample for quick view
            "sample_masked": [r["masked"] for r in results[:200]],
        }
        await checks_collection.insert_one(metadata)
    except Exception:
        logger.exception("Failed to log check metadata in Mongo.")

    # cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    try:
        os.unlink(report_tmp_path)
    except Exception:
        pass


# ---------------- OWNER / SUDO MANAGEMENT ----------------
async def addsudo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caller = update.effective_user
    if not await is_owner(caller.id):
        await update.message.reply_text("🚫 Only owners can add sudo users.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /addsudo <user_id>")
        return
    try:
        target = int(context.args[0])
        await add_sudo_db(target, caller.id)
        await update.message.reply_text(f"✅ Added {target} as sudo.")
    except ValueError:
        await update.message.reply_text("Invalid user id.")


async def removesudo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caller = update.effective_user
    if not await is_owner(caller.id):
        await update.message.reply_text("🚫 Only owners can remove sudo users.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /removesudo <user_id>")
        return
    try:
        target = int(context.args[0])
        await remove_sudo_db(target)
        await update.message.reply_text(f"✅ Removed {target} from sudo.")
    except ValueError:
        await update.message.reply_text("Invalid user id.")


async def listsudo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not await is_sudo(user.id):
        await update.message.reply_text("🚫 Only owners/sudo can list sudo users.")
        return
    users = await list_sudo_db()
    if not users:
        await update.message.reply_text("No sudo users.")
        return
    msg = "Sudo users:\n" + "\n".join([f"- {u['user_id']} (added_by: {u.get('added_by')})" for u in users])
    await update.message.reply_text(msg)


# export last N checks metadata (owner-only)
async def exportchecks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caller = update.effective_user
    if not await is_owner(caller.id):
        await update.message.reply_text("🚫 Only owners can export check metadata.")
        return
    n = 50
    if context.args:
        try:
            n = int(context.args[0])
        except ValueError:
            pass
    cursor = checks_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(n)
    items = [doc async for doc in cursor]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    json.dump(items, tmp, default=str, indent=2)
    tmp.close()
    try:
        await update.message.reply_document(InputFile(tmp.name, filename=f"checks_export_{n}.json"))
    except Exception:
        logger.exception("Failed to send checks export")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ---------------- BOT SETUP ----------------
def main():
    if BOT_TOKEN == "PUT_YOUR_BOT_TOKEN_HERE":
        logger.error("Please set BOT_TOKEN in the script before running.")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    # public commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("check", check_cmd))

    # bulk and document handling (sudo-only enforced in handler)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # owner/sudo management
    app.add_handler(CommandHandler("addsudo", addsudo_cmd))
    app.add_handler(CommandHandler("removesudo", removesudo_cmd))
    app.add_handler(CommandHandler("listsudo", listsudo_cmd))
    app.add_handler(CommandHandler("exportchecks", exportchecks_cmd))

    logger.info("Starting bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
