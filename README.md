# Luhn CC Checker — Telegram Bot

A lightweight Telegram bot that validates credit/debit card lines using the Luhn algorithm, basic BIN pattern checks, expiry and CVV validation, and produces masked reports. It supports single checks via a command and bulk checks from uploaded files, with optional reporting to a channel and metadata logging in MongoDB.

> Important: This project is for educational/testing purposes only. Do not use it on real payment card data without proper authorization and compliance. The bot masks card numbers in reports and omits CVVs by default.

## Features
- Luhn check and card type detection (Visa, MasterCard, AmEx, Discover, Diners, JCB)
- Expiry and CVV validation (length rules)
- Telegram bot commands: `/start`, `/help`, `/check`, sudo management, and export
- Bulk checking from uploaded `.txt`, `.pdf`, or `.docx` files (sudo/owner only)
- Masked report generation with LIVE/DEAD simulation (configurable probability)
- Optional posting of masked reports to a channel
- MongoDB logging of check metadata (no raw CVV stored)
- Simple per-user rate limiting for bulk operations

## Requirements
- Python 3.9+ (tested with 3.10/3.11)
- MongoDB (local or remote)
- A Telegram Bot API token from BotFather
- Windows PowerShell (instructions below) or any shell

Python packages are listed in `requirements.txt`.

## Quick Start (Windows PowerShell)

1) Clone or open the project folder.

2) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4) Make sure MongoDB is running. Options:
- Local install: install MongoDB Community Server and ensure it listens on `mongodb://localhost:27017`.
- Or Docker:

```powershell
docker run -d --name mongo -p 27017:27017 mongo:6.0
```

5) Configure the bot (env vars are supported):
- Set environment variables in your shell:

```powershell
$env:BOT_TOKEN = "123456:ABC-YourBotToken"
$env:MONGO_URI = "mongodb://localhost:27017"
$env:DB_NAME = "cc_checker_bot"
```

- Or configure directly in `checker.py` (constants at the top) if you prefer. Also set:
	- `OWNER_IDS` (list of Telegram numeric user IDs allowed to add/remove sudo).
	- `CHANNEL_ID` (numeric chat ID of a channel, e.g. `-100xxxxxxxxxxxx`; optional if you don’t want channel reports).
	- Optionally adjust: `LIVE_PROBABILITY`, `MAX_FILE_SIZE_MB`, `RATE_LIMIT_SECONDS`, `STORE_FULL_RAW_LINES`, `MASK_REVEAL_LAST`.

6) Run the bot:

```powershell
python checker.py
```

The bot uses long polling. Talk to your bot in Telegram once it’s running.

## Usage

- `/start` — welcome message.
- `/help` — available commands.
- `/check <number|MM|YYYY|CVV>` — validate a single line, e.g.:

```
/check 4111111111111111|12|2027|123
```

- Bulk checking (sudo/owner only): upload a `.txt`, `.pdf`, or `.docx` file. The bot parses each line and replies with a masked report file.

Accepted line format (per line):
```
<number>|<MM>|<YYYY>|<CVV>
```
Examples: `4111111111111111|12|2027|123` or `5555555555554444|01|2026|456`.

Notes:
- Only masked numbers are used in reports; CVVs are not included.
- Bulk checks are rate limited per user (`RATE_LIMIT_SECONDS`).
- Report optionally posts to the configured `CHANNEL_ID`.

## Permissions and Roles
- Owners: user IDs listed in `OWNER_IDS` inside `checker.py`.
- Sudo: stored in MongoDB collection `sudo_users`. Owners can add/remove sudo users.

Commands:
- `/addsudo <user_id>` (owner only)
- `/removesudo <user_id>` (owner only)
- `/listsudo` (sudo/owner)
- `/exportchecks <N>` (owner only) — export last N metadata entries as JSON.

## Data and Privacy
- By default, raw lines are NOT stored (`STORE_FULL_RAW_LINES = False`).
- Logged metadata (collection `checks`) includes counts and up to 200 masked samples for quick view.
- You should review the code and adjust data retention to your policies.

## Configuration Tips
- Getting a channel ID: add your bot to the channel, send a test message, and use a bot like `@RawDataBot` or other tools to retrieve the numeric `chat.id`. It usually looks like `-100...`.
- If you see an import error similar to `cannot import name ChatAction` with `python-telegram-bot` v20+, change the import to `from telegram.constants import ChatAction` in `checker.py`.

## Troubleshooting
- Bot token invalid: ensure the `BOT_TOKEN` from BotFather is correct.
- Bot not responding: check that the process is running and there are no errors in the console; ensure your machine has internet access.
- Mongo connection fails: verify `MONGO_URI`, that MongoDB is running, and network access is allowed.
- PDF/DOCX parsing errors: ensure `PyMuPDF` and `python-docx` installed; try with a `.txt` file to isolate parsing issues.
- File too large: adjust `MAX_FILE_SIZE_MB` in `checker.py`.

## Development
- Run with an IDE/VS Code and use breakpoints.
- Logging uses Python’s `logging` module; adjust log level in `checker.py`.

## Legal & Ethical Notice
This software is for lawful, authorized testing, education, and demonstration only. You are responsible for compliance with applicable laws, regulations, and cardholder data protection standards (e.g., PCI DSS). Never collect, process, or store real cardholder data without proper authorization and controls.
