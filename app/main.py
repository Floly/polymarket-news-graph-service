import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from pathlib import Path
import asyncio
from .data_collection import DataCollector
from .news_parser import NewsParser
from .nlp_processing import NLPProcessor
from .graph_builder import GraphBuilder
from .inference import InferenceEngine
from .storage import Storage
# from .monitoring import Monitoring
from .config import Config

# FastAPI app
app = FastAPI()

# Handle favicon.ico requests
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)  # No Content

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic model for webhook payload
class TelegramWebhook(BaseModel):
    update: dict

# Pydantic model for predict endpoint
class PredictRequest(BaseModel):
    url: str

# Initialize components
data_collector = DataCollector()
news_parser = NewsParser()
nlp_processor = NLPProcessor()
graph_builder = GraphBuilder()
inference_engine = InferenceEngine()
storage = Storage()
# monitoring = Monitoring()

async def predict_backbone(request):
    try:
        # Process the request
        event_data = await data_collector.fetch_event_data(request.url)
        event_id = event_data['id']
        storage.log_request(event_id, request.url)
        
        entities = nlp_processor.process_event(event_data)
        storage.save_entities(event_id, entities)

        # Parse news
        articles = await news_parser.parse_news(event_data)
        storage.save_articles(event_id, articles)
        
        # NLP processing
        embeddings = nlp_processor.process_articles(articles)
        storage.save_embeddings(event_id, embeddings)
        
        # Build graph
        graph = graph_builder.build_graph(event_data, articles, embeddings)
        storage.save_graph(event_id, graph)
        
        # Run inference
        predictions = inference_engine.run_inference(event_data, graph)
        storage.save_predictions(event_id, predictions)
        # monitoring.log_metrics(event_id, predictions)
        
        # Format response as JSON
        return {
            "event_title": event_data['title'],
            "predictions": [
                {
                    "market": market['question'],
                    "graph_prediction": predictions['graph'][i],
                    "price_prediction": predictions.get('price', {}).get(market['id'], 0.0)
                }
                for i, market in enumerate(event_data['markets'][:4])
            ]
        }
    except Exception as e:
        logger.error(f"Error processing predict request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Telegram bot handlers
async def start(update: Update, context):
    await update.message.reply_text("Send a Polymarket event URL to get predictions.")

async def handle_message(update: Update, context):
    url = update.message.text
    await predict_backbone(url)


# New FastAPI endpoint for direct predictions
@app.post("/predict")
async def predict(request: PredictRequest):
    await predict_backbone(request)

# Initialize Telegram bot
bot = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
bot.add_handler(CommandHandler("start", start))
bot.add_handler(MessageHandler(filters.Text() & ~filters.Command(), handle_message))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))