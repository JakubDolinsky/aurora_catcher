import heapq
import os

import torch

from application import config
from application.cnn_layer.models.model2.cnn_model import AuroraCNN
from application.common.image_preprocess import preprocess_img

BASE_DIR= os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights", "best_model.pt")

IMAGE_SQUARE_SIZE = 256

class Model2InferenceEngine:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AuroraCNN()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def classify_level(self, val, probability_levels):
        medium, high = probability_levels[1:]

        if val < medium:
            return "low"
        elif val < high:
            return "medium"
        else:
            return "high"

    def infer(self, img_path):
        with (torch.inference_mode()):
            img_to_infer = preprocess_img(img_path).unsqueeze(0).to(self.device)
            logits = self.model(img_to_infer)
            probs = torch.sigmoid(logits).cpu().squeeze(0)
            probs_classes_dict = {}
            for i in range(len(config.CLASS_NAMES)):
                probs_classes_dict[config.CLASS_NAMES[i]] = probs[i].item()

            top_items = heapq.nlargest(
                3,
                (
                    (k, v)
                    for k, v in probs_classes_dict.items()
                    if k in config.INFER_DECISION_THRESHOLDS_DICT
                       and v >= config.INFER_DECISION_THRESHOLDS_DICT[k]
                ),
                key=lambda item: item[1]
            )

            result = {
                k: {
                    "value": v,
                    "level": self.classify_level(v, config.INFER_DECISION_PROBABILITY_LEVELS_DICT[k])
                }
                for k, v in top_items
            }
            message = f"Model 2 inference values: {result}"
            return message, result




