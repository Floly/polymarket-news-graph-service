import logging
import json
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import gc

logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self):
        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
            self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            self.bert_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            self.bert_ner_pipeline = pipeline("ner", model=self.bert_model, tokenizer=self.tokenizer, aggregation_strategy="first")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
            raise

    def process_event(self, event, articles):
        event_id = event['id']
        event_path = f"../data/inference_data/{event_id}"
        
        # Extract entities
        event_description = event.get("description", "")
        bert_entities = list({
            ent["word"].lower()
            for ent in self.bert_ner_pipeline(event_description)
            if not self.spacy_nlp(ent["word"].lower())[0].is_stop
        }) if event_description else []
        
        tags = event.get("tags", [])
        tag_labels = list({tag.get("label", "").lower() for tag in tags if tag.get("label")})
        tag_slugs = list({tag.get("slug", "") for tag in tags if tag.get("slug")})
        
        m_ents = {}
        for market in event.get("markets", []):
            market_id = market.get("id")
            market_question = market.get("question", "")
            m_ents[market_id] = market_question
        
        event_entities = {
            event_id: {
                "event_title": event.get("title"),
                "event_ents_bert": bert_entities,
                "tag_labels": tag_labels,
                "tag_slugs": tag_slugs,
                "market_ents": m_ents
            }
        }
        
        # Generate embeddings for articles
        embeddings = {}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/"
        }
        
        for article in articles:
            url = article.get('real_url', '')
            article_id = article.get('id', '')
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.content, 'html.parser')
                sentences = self.extract_sentences(soup.text)
                if not sentences:
                    continue
                embeddings[article_id] = self.sentence_model.encode(sentences)
                del soup, response
                gc.collect()
            except Exception as e:
                logger.warning(f"Error processing article {article_id}: {e}")
        
        return event_entities, embeddings

    def normalize_unicode_text(self, text):
        try:
            return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        except Exception as e:
            logger.warning(f"Error normalizing Unicode text: {e}")
            return text

    def clean_text(self, text):
        try:
            text = re.sub(r'\s+', ' ', text).strip()
            text = self.normalize_unicode_text(text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'\d+', '', text)
            text = text.lower()
            return text
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return ""

    def extract_sentences(self, text):
        try:
            sentences = self.clean_text(text).split('.')
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except Exception as e:
            logger.warning(f"Error extracting sentences: {e}")
            return []