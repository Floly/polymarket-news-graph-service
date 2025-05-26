import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
from data_collection import DataCollector
from news_parser import NewsParser
from nlp_processing import NLPProcessor
from graph_builder import GraphBuilder
from inference import InferenceEngine
from storage import Storage
from monitoring import Monitoring
from config import Config

# FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic model for webhook payload
class TelegramWebhook(BaseModel):
    update: dict

# Initialize components
data_collector = DataCollector()
news_parser = NewsParser()
nlp_processor = NLPProcessor()
graph_builder = GraphBuilder()
inference_engine = InferenceEngine()
storage = Storage()
monitoring = Monitoring()

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send a Polymarket event URL to get predictions.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text
    try:
        # Process the request
        event_data = await data_collector.fetch_event_data(url)
        event_id = event_data['id']
        storage.log_request(event_id, url)
        
        # Parse news
        articles = await news_parser.parse_news(event_data)
        storage.save_articles(event_id, articles)
        
        # NLP processing
        entities, embeddings = nlp_processor.process_event(event_data, articles)
        storage.save_entities(event_id, entities)
        storage.save_embeddings(event_id, embeddings)
        
        # Build graph
        graph = graph_builder.build_graph(event_data, articles, embeddings)
        storage.save_graph(event_id, graph)
        
        # Run inference
        predictions = inference_engine.run_inference(event_data, graph)
        storage.save_predictions(event_id, predictions)
        monitoring.log_metrics(event_id, predictions)
        
        # Format response
        response = format_predictions(event_data, predictions)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        await update.message.reply_text("An error occurred. Please try again.")
        monitoring.log_error(event_id, str(e))

def format_predictions(event_data, predictions):
    output = f"Predictions for event: {event_data['title']}\n"
    for i, market in enumerate(event_data['markets']):
        graph_pred = predictions['graph'][i]
        price_pred = predictions.get('price', {}).get(market['id'], 0.0)
        output += f"Market: {market['question']}\nGraph Prediction: {graph_pred:.4f}\nPrice Prediction: {price_pred:.4f}\n\n"
    return output

# FastAPI endpoint for Telegram webhook
@app.post("/webhook")
async def webhook(webhook_data: TelegramWebhook):
    update = Update.de_json(webhook_data.update, bot)
    await dp.process_update(update)
    return {"status": "ok"}

# Initialize Telegram bot
bot = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
dp = bot.dispatcher
dp.add_handler(CommandHandler("start", start))
dp.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))