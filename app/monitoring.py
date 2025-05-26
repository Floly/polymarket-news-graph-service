import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Monitoring:
    def __init__(self):
        self.metrics_path = "../logs/metrics.json"
        self.metrics = self.load_metrics()

    def load_metrics(self):
        try:
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_metrics(self):
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def log_metrics(self, event_id, predictions):
        timestamp = datetime.now().isoformat()
        self.metrics[event_id] = self.metrics.get(event_id, {})
        self.metrics[event_id][timestamp] = {
            'graph_predictions': predictions['graph'],
            'price_predictions': predictions['price']
        }
        self.save_metrics()
        logger.info(f"Logged metrics for event {event_id}")

    def log_error(self, event_id, error):
        timestamp = datetime.now().isoformat()
        self.metrics[event_id] = self.metrics.get(event_id, {})
        self.metrics[event_id][timestamp] = {'error': error}
        self.save_metrics()
        logger.error(f"Logged error for event {event_id}: {error}")