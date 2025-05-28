import logging
import os
import json
import networkx as nx
from itertools import combinations
from transformers import BertTokenizer, BertForTokenClassification, pipeline

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self):
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")
            self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        except Exception as e:
            logger.error(f"Failed to load NER pipeline: {e}")
            raise

    def build_graph(self, event, articles, embeddings):
        event_id = event['id']
        embeddings_path = f"data/inference_data/{event_id}/sentence_embeddings/"
        
        embedded_article_ids = [x.split('.')[0] for x in os.listdir(embeddings_path)]
        valid_articles = []
        for article in articles:
            aid = article.get('id')
            if aid in embedded_article_ids:
                try:
                    article['ner'] = self.extract_entities(article['title'])
                    valid_articles.append(article)
                except Exception as e:
                    logger.warning(f"Error processing article {aid}: {e}")
        
        if not valid_articles:
            logger.warning(f"No valid articles found for event {event_id}")
            return None
        
        G = nx.Graph()
        for item in valid_articles:
            G.add_node(item['id'], article=item['real_url'])
        
        for a, b in combinations(valid_articles, 2):
            id_a, id_b = a['id'], b['id']
            same_date = a['date'] == b['date']
            same_query = a['query'] == b['query']
            same_publisher = a['publisher'] == b['publisher']
            common_entities = a['ner'].intersection(b['ner'])
            common_entities_share = len(common_entities) / max(len(a['ner']) or 1, len(b['ner']) or 1)
            
            if any([same_date, same_query, same_publisher, common_entities_share > 0.35]):
                G.add_edge(
                    id_a,
                    id_b,
                    same_date=same_date,
                    same_query=same_query,
                    same_publisher=same_publisher,
                    common_entities_share=common_entities_share
                )
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def extract_entities(self, title):
        try:
            ner_results = self.ner_pipeline(title)
            return set(entity['word'] for entity in ner_results if entity['entity'].startswith('B-'))
        except Exception as e:
            logger.warning(f"Error extracting entities from title: {e}")
            return set()