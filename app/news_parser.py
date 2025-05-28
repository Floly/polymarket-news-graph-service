import logging
import json
import asyncio
import random
from itertools import combinations
from datetime import datetime
from tqdm import tqdm
from .utils.parsers import fetch_google_news_rss, get_real_url_async

logger = logging.getLogger(__name__)

class NewsParser:
    MAX_SAMPLE_SIZE = 20

    async def parse_news(self, event):
        event_id = event['id']
        entities_path = f"data/inference_data/{event_id}/entities.json"
        
        with open(entities_path, 'r') as f:
            event_ents = json.load(f)
        
        markets = event['markets']
        last_date_str = datetime.now().strftime('%Y-%m-%d')
        
        tags = event_ents[event_id]['tag_labels']
        tag_combinations = list(combinations(tags, len(tags)))
        tag_queries = [' '.join(combo) for combo in tag_combinations]
        
        queries = []
        queries.extend(event_ents[event_id]['event_ents_bert'])
        queries.extend(tag_queries)
        queries.extend([m.get('question') for m in markets])
        
        news_data = self.fetch_news_for_queries(queries, last_date_str)
        
        articles = [
            {**article, 'query': source}
            for source, articles_list in news_data.items()
            for article in articles_list
        ]
        
        articles = await self.url_extractor(event_id, articles)
        return articles

    def fetch_news_for_queries(self, queries, cutoff_date):
        logger.info(f"Fetching news for {len(queries)} queries (cutoff: {cutoff_date})")
        news_data = {}
        for query in tqdm(queries):
            try:
                res = fetch_google_news_rss(query, cutoff_date=cutoff_date)
                if res:
                    news_data.update(res)
            except Exception as e:
                logger.error(f"Error fetching news for query '{query}': {e}")
        return news_data

    async def url_extractor(self, event_id, articles, num_parallel_tasks=20):
        logger.info(f"Processing {len(articles)} articles for event ID: {event_id}")
        
        if len(articles) > self.MAX_SAMPLE_SIZE:
            logger.warning(f"Exceeded {self.MAX_SAMPLE_SIZE} articles. Sampling {self.MAX_SAMPLE_SIZE}.")
            articles = random.sample(articles, self.MAX_SAMPLE_SIZE)
        
        chunk_size = max(1, len(articles) // num_parallel_tasks)
        chunks = [articles[i:i + chunk_size] for i in range(0, len(articles), chunk_size)]
        
        tasks = [get_real_url_async(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        combined_articles = []
        for result in results:
            combined_articles.extend(result)
        
        return combined_articles