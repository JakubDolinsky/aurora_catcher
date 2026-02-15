import os

import torch

from application import config
from application.cnn_layer.models.model1.cnn_model import AuroraCNN
from application.common.image_preprocess import preprocess_img

BASE_DIR= os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights", "best_model.pt")

IMAGE_SQUARE_SIZE = 256
T_FALSE = config.SIGMOID_INFERENCE_THRESHOLDS[0]
T_TRUE = config.SIGMOID_INFERENCE_THRESHOLDS[1]

class Model1InferenceEngine:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AuroraCNN()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def infer(self, img_path):
        with (torch.inference_mode()):
            img_to_infer = preprocess_img(img_path).unsqueeze(0).to(self.device)
            logit = self.model(img_to_infer)
            prob = torch.sigmoid(logit).squeeze(1)[0]
            #message is for reporting in console and log file
            if prob <= T_FALSE:
                message = f"prob = {prob:.3f}, decision = false"
                return message, 0
            elif prob >= T_TRUE:
                message = f"prob = {prob:.3f}, decision = true"
                return message, 1
            else:
                message = f"prob = {prob:.3f}, decision = uncertain"
                return message, -1





