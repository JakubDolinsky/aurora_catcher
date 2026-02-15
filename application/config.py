CLASS_NAMES = ["airglow", "light pollution", "lightning", "milky way", "NLC", "twilight", "zodiacal light"]
ALLOW_LOGGING = False
#inference
#model1 thresholds
SIGMOID_INFERENCE_THRESHOLDS = (0.15, 0.75)
#model2 thresholds
INFER_DECISION_THRESHOLDS_DICT = {
    "airglow":0.3600,
    "light pollution":0.3200,
    "lightning":0.3800,
    "milky way":0.5000,
    "NLC":0.4600,
    "twilight":0.2600,
    "zodiacal light":0.4100
}

#probability points for low, medium and high probability obtained in script adjust_positive_threshold_and_prob_levels.py
INFER_DECISION_PROBABILITY_LEVELS_DICT = {
    "airglow":[0.3600, 0.9342, 0.9795],
    "light pollution":[0.3200, 0.8194, 0.9368],
    "lightning":[0.3800, 0.9864, 0.9971],
    "milky way":[0.5000, 0.9193, 0.9692],
    "NLC":[0.4600, 0.9961, 0.9984],
    "twilight":[0.2600, 0.9767, 0.9942],
    "zodiacal light":[0.4100, 0.7018, 0.8013]
}





