import os

class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your-telegram-bot-token")
    PORT = int(os.getenv("PORT", 8000))