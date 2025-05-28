import sys
import logging
import torch
import numpy as np
import joblib
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from sentence_transformers import SentenceTransformer
from tsfresh import extract_features
from .utils.gcn_v2 import GCNGraphClassifier_v2


sys.path.insert(0, 'app/')

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.gcn_model = GCNGraphClassifier_v2(13, 64)

        state_dict = torch.load('models/GCNGraphClassifier_v2_20250517_164339.pth')
        self.gcn_model.load_state_dict(state_dict)
        self.gcn_model.eval()
        with open('models/price_predictor.pkl', 'rb') as f:
            self.price_clf = joblib.load(f)

    def run_inference(self, event, graph):
        graphs = []
        for market in event['markets']:
            try:
                g = self.generate_graph(market, graph, self.sentence_transformer)
                graphs.append(g)
            except Exception as e:
                logger.warning(f"Error processing market: {e}")
                continue
        
        graph_data = self.add_key(graphs, 'same_date')
        dataloader = DataLoader(graph_data, batch_size=1)
        
        predictions = {'graph': [], 'price': {}}
        for data in dataloader:
            out = self.gcn_model(data)
            pred = torch.exp(out)[:, 0]
            predictions['graph'].append(pred.item())
        
        for i, market in enumerate(event['markets']):
            try:
                X = self.series_to_tsfresh_df(market['prices_history'][0]['history'], series_id=0)
                X = extract_features(X, column_id='id', column_sort='time', column_value='value')
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                price_pred = self.price_clf.predict_proba(X)[:, 1].item()
                predictions['price'][market['id']] = price_pred
            except Exception as e:
                logger.warning(f"Error predicting price for market {market['id']}: {e}")
        
        return predictions

    def generate_graph(self, market, G, model):
        pos = np.argmax(eval(market['outcomePrices']))
        y = np.where(eval(market['outcomes'])[pos] == 'No', 0, 1)
        market_question = market['question']
        question_emb = model.encode(market_question)
        
        for node_id in G.nodes:
            article_emb = np.load(f'data/inference_data/{market["event_id"]}/sentence_embeddings/{node_id}.npz')
            node_vector = self.get_similarity_vector(article_emb, question_emb, model)
            G.nodes[node_id]['embedding'] = node_vector
        
        graph = from_networkx(G)
        graph.y = torch.tensor(y)
        return graph

    def get_similarity_vector(self, article_emb, question_emb, model):
        similarities_list = [
            model.similarity(s_emb, question_emb)
            for s_emb in article_emb['arr_0']
        ]
        similarities = torch.cat(similarities_list)
        idx = similarities.argsort(dim=0, descending=True)[:10]
        article_similarity = torch.cat([
            torch.quantile(similarities, torch.tensor([i/10.0 for i in range(10)]), dim=0, keepdim=False).flatten(),
            torch.tensor([similarities.mean(), max(0, similarities[idx].std()), len(similarities > 0.5)])
        ])
        return article_similarity

    def add_key(self, graph_list, key):
        for g in graph_list:
            try:
                g[key]
            except KeyError:
                g[key] = torch.zeros(g.edge_index.size()[1])
        graph_list_new = [
            Data(x=g['embedding'], y=g['y'], edge_index=g['edge_index'], same_date=g[key])
            for g in graph_list if g is not None
        ]
        return graph_list_new

    def series_to_tsfresh_df(self, history, series_id):
        return pd.DataFrame({
            'id': [series_id for _ in history],
            'time': [point['t'] for point in history],
            'value': [point['p'] for point in history]
        })