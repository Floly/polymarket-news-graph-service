import logging
import json
import networkx as nx
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class Storage:
    def __init__(self):
        self.root = "data/inference_data/"

    def log_request(self, event_id, url):
        logger.info(f"Received request for event {event_id}: {url}")

    def save_articles(self, event_id, articles):
        path = f"{self.root}{event_id}/articles.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(articles, f)
        logger.info(f"Saved articles to {path}")

    def save_entities(self, event_id, entities):
        path = f"{self.root}{event_id}/entities.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(entities, f, indent=2)
        logger.info(f"Saved entities to {path}")

    def save_embeddings(self, event_id, embeddings):
        embeddings_path = f"{self.root}{event_id}/sentence_embeddings/"
        Path(embeddings_path).mkdir(parents=True, exist_ok=True)
        for article_id, emb in embeddings.items():
            with open(f"{embeddings_path}{article_id}.npz", 'wb') as f:
                np.savez_compressed(f, emb)
        logger.info(f"Saved embeddings to {embeddings_path}")

    def save_graph(self, event_id, graph):
        path = f"{self.root}{event_id}/graph.graphml"
        if graph:
            nx.write_graphml(graph, path)
            logger.info(f"Saved graph to {path}")

    def save_predictions(self, event_id, predictions):
        path = f"{self.root}{event_id}/predictions.json"
        with open(path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved predictions to {path}")