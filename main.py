import asyncio
import html
import logging
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any
import time
from collections import deque

from dotenv import load_dotenv
from google import genai
from telegram import Update
from telegram.error import BadRequest
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# load .env
load_dotenv(Path(__file__).with_name(".env"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

DB_PATH = Path(os.getenv("SETTINGS_DB_PATH", Path(__file__).with_name("bot_settings.db")))

# Pricing is configurable because model prices can change.
# Values are USD per 1M tokens.
INPUT_COST_PER_1M_TOKENS = float(os.getenv("INPUT_COST_PER_1M_TOKENS", "0"))
OUTPUT_COST_PER_1M_TOKENS = float(os.getenv("OUTPUT_COST_PER_1M_TOKENS", "0"))
# Optional fallback estimate if 1M token rates are not configured.
FALLBACK_COST_PER_1K_TOKENS = float(os.getenv("FALLBACK_COST_PER_1K_TOKENS", "0"))

MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
RATE_LIMIT_COUNT = 5
RATE_LIMIT_WINDOW = 30  # seconds
chat_request_log: dict[int, deque[float]] = {}

PROMPT_TEMPLATE = """Rewrite the following text in a clear, professional tone.
Do not use HTML, Markdown, or any formatting symbols (except emojis if specifically requested by the user).
Do not add explanations.
Preserve line breaks.
Original text:
{user_text}"""

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

def validate_env() -> None:
    missing = [
        name
        for name, value in {
            "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_settings (
                chat_id INTEGER PRIMARY KEY,
                cost_enabled INTEGER NOT NULL DEFAULT 0
            )
            """
        )

def get_cost_enabled(chat_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT cost_enabled FROM chat_settings WHERE chat_id = ?", (chat_id,)
        ).fetchone()
    return bool(row[0]) if row else False

def set_cost_enabled(chat_id: int, enabled: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO chat_settings(chat_id, cost_enabled)
            VALUES(?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET cost_enabled = excluded.cost_enabled
            """,
            (chat_id, int(enabled)),
        )

def generate_rewrite(user_text: str) -> Any:
    if not GEMINI_API_KEY:
        # Mock response for testing
        class FakeResponse:
            text = f"TEST MODE:\n{user_text}"
            usage_metadata = None
        return FakeResponse()

    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=PROMPT_TEMPLATE.format(user_text=user_text),
    )

def is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_signals = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "deadline exceeded",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
    ]
    return any(signal in text for signal in transient_signals)

async def generate_rewrite_with_retry(user_text: str) -> Any:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(generate_rewrite, user_text)
        except Exception as exc:
            if attempt >= MAX_RETRIES or not is_transient_error(exc):
                raise
            sleep_seconds = (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            logger.warning(
                "Gemini transient error (attempt %s/%s): %s",
                attempt,
                MAX_RETRIES,
                exc,
            )
            await asyncio.sleep(sleep_seconds)
    raise RuntimeError("Gemini call failed after retries")

def _read_usage_attr(obj: Any, attr: str, default: int = 0) -> int:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return int(obj.get(attr, default) or default)
    return int(getattr(obj, attr, default) or default)

def extract_usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = getattr(response, "usageMetadata", None)

    prompt_tokens = _read_usage_attr(usage, "prompt_token_count", 0)
    completion_tokens = _read_usage_attr(usage, "candidates_token_count", 0)
    total_tokens = _read_usage_attr(usage, "total_token_count", prompt_tokens + completion_tokens)

    if total_tokens == 0:
        metadata = getattr(response, "metadata", None) or {}
        token_usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
        prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens

def estimate_cost(prompt_tokens: int, completion_tokens: int, total_tokens: int) -> float:
    if INPUT_COST_PER_1M_TOKENS > 0 or OUTPUT_COST_PER_1M_TOKENS > 0:
        return (
            (prompt_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS
            + (completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
        )

    if FALLBACK_COST_PER_1K_TOKENS > 0:
        return (total_tokens / 1000) * FALLBACK_COST_PER_1K_TOKENS

    return 0.0

async def toggle_cost(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    chat_id = update.message.chat_id
    current = get_cost_enabled(chat_id)
    new_status = not current
    set_cost_enabled(chat_id, new_status)

    status = "ON" if new_status else "OFF"
    await update.message.reply_text(f"Cost display is now {status}.")

async def rewrite(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return

    # Start typing indicator
    await context.bot.send_chat_action(
        chat_id=update.message.chat_id, action=ChatAction.TYPING
    )

    user_text = update.message.text
    chat_id = update.message.chat_id

    now = time.time()
    log = chat_request_log.setdefault(chat_id, deque())

    # remove old timestamps outside the window
    while log and now - log[0] > RATE_LIMIT_WINDOW:
        log.popleft()

    if len(log) >= RATE_LIMIT_COUNT:
        await update.message.reply_text(
            "You're sending messages too quickly. Please wait a few seconds and try again."
        )
        return

    log.append(now)

    try:
        response = await generate_rewrite_with_retry(user_text)

        rewritten = getattr(response, "text", "") or ""
        if not rewritten:
            rewritten = str(response)
        clean_text = re.sub(r"<[^>]+>", "", rewritten)
        for chunk_start in range(0, len(clean_text), 4000):
            await update.message.reply_text(clean_text[chunk_start:chunk_start+4000])

        if get_cost_enabled(chat_id):
            prompt_tokens, completion_tokens, total_tokens = extract_usage(response)
            estimated_cost = estimate_cost(prompt_tokens, completion_tokens, total_tokens)

            await update.message.reply_text(
                (
                    f"<b>Tokens used:</b> {total_tokens}\n"
                    f"<b>Input tokens:</b> {prompt_tokens}\n"
                    f"<b>Output tokens:</b> {completion_tokens}\n"
                    f"<b>Estimated cost:</b> ${estimated_cost:.6f}"
                ),
                parse_mode="HTML",
            )

    except Exception:
        logger.exception("Failed to rewrite message for chat_id=%s", chat_id)
        await update.message.reply_text(
            "I couldn't rewrite this message right now. Please try again in a moment."
        )

def main() -> None:
    validate_env()
    init_db()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, rewrite))
    app.add_handler(CommandHandler("togglecost", toggle_cost))

    logger.info("Bot is live with Gemini model=%s", GEMINI_MODEL)
    app.run_polling()

if __name__ == "__main__":
    main()
