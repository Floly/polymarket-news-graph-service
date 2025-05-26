import logging
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime, timedelta
from utils.parsers import PolymarketAPIClient, PriceHistoryFetcher, fetch_google_news_rss
from utils.base import DateConverter

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.event_api = PolymarketAPIClient("https://gamma-api.polymarket.com/events")
        self.price_api = PriceHistoryFetcher(PolymarketAPIClient("https://clob.polymarket.com/prices-history/"))
        self.root = "../data/inference_data/"

    async def fetch_event_data(self, url):
        o = urlparse(url)
        slug = o.path.split('/')[2]
        res = self.event_api.get_events(params={'slug': slug})
        if not res:
            raise ValueError("Failed to fetch event data")
        
        event = res[0]
        event_id = event['id']
        event_path = f'{self.root}{event_id}'
        Path(f'{event_path}/sentence_embeddings/').mkdir(parents=True, exist_ok=True)
        
        # Configure logging for this event
        handler = logging.FileHandler(f'../logs/{event_id}.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        # Fetch price history
        now = datetime.now()
        end_ts = now.timestamp()
        start_ts = max(
            (now - timedelta(days=14)).timestamp(),
            DateConverter().iso_or_yy_mm_dd_to_unix(event['markets'][0]['startDate'])
        )
        
        for market in event['markets']:
            try:
                tid = market['clobTokenIds']
                prices = []
                for market_id in eval(tid):
                    price = self.price_api.fetch_prices_history(
                        market_id=market_id,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        fidelity=60
                    )
                    prices.append(price)
                market['prices_history'] = prices
            except KeyError:
                continue
        
        return event