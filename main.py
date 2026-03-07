import os
import re
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler
from google import genai

# load .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Track cost display per chat
cost_toggle = {}  # chat_id: True/False

async def toggle_cost(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    current = cost_toggle.get(chat_id, False)
    cost_toggle[chat_id] = not current
    status = "ON" if cost_toggle[chat_id] else "OFF"
    await update.message.reply_text(f"Cost display is now {status}.")

async def rewrite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.message.chat_id

    try:
        # Call Gemini to rewrite text
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Rewrite the following text in a clear, professional tone.
            All emphasis must use HTML tags only:
            - Use <b>…</b> for bold
            - Use <i>…</i> for italics
            - Do NOT use Markdown (*…* or **…**) or any other symbols for emphasis
            - Do NOT add explanations
            - Preserve line breaks
            Original text:
            {user_text}"""
        )

        # The text property holds the generated rewrite
        rewritten = response.text
        
        await update.message.reply_text(rewritten, parse_mode="HTML")

        # Send cost info only if toggle is ON
        if cost_toggle.get(chat_id, False):
            usage_info = getattr(response, "metadata", {}).get("token_usage", {})
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            rate_per_1k_tokens = 0.001  # adjust per your plan
            estimated_cost = total_tokens * rate_per_1k_tokens

            await update.message.reply_text(
                f"<b>Tokens used:</b> {total_tokens}\n<b>Estimated cost:</b> ${estimated_cost:.4f}",
                parse_mode="HTML"
            )

    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

# Telegram bot setup
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, rewrite))
app.add_handler(CommandHandler("togglecost", toggle_cost))

print("Bot is live with Gemini!")
app.run_polling()
