from application.cnn_layer.model1_inference_engine.model1_inference_engine import Model1InferenceEngine
from application.cnn_layer.model2_inference_engine.model2_inference_engine import Model2InferenceEngine

class MidLayer:
    def __init__(self):
        self.model1_inference_engine = Model1InferenceEngine()
        self.model2_inference_engine = Model2InferenceEngine()

    def decide_if_aurora_or_detect_other_phenomena(self, img_path):
        results ={}
        result1 = self.model1_inference_engine.infer(img_path)
        results["aurora_decision"] = result1[1]
        results["probability"] = result1[0]
        if result1[1] != 1:
            result2 = self.model2_inference_engine.infer(img_path)
            results["secondary_phenomena"] = dict(result2[1])
        return results
