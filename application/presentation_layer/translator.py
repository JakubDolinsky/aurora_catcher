#model1 result messages
import datetime

from application.common.logger import log
AURORA_DETECTION_VERDICT = ("AURORA DETECTED!", "AURORA NOT DETECTED!", "AURORA NOT CERTAIN!")
AURORA_EXISTS = "There is high probability of aurora occurrence in the picture.\n"
NO_AURORA_EXISTS = "There is very low or no probability of aurora occurrence in the picture.\n"
AURORA_UNCERTAIN = ("It is not possible to confirm aurora occurrence in the picture with reasonable reliability."
                    "Try another picture\n")

#model2 result messages
AURORALIKE_PHENOMENA_EXISTS = "Phenomena similar to aurora has been detected. List of detected phenomena with probability value and level below: \n"
NO_AURORALIKE_PHENOMENA_EXISTS = "No phenomena similar to aurora has been detected in the picture.\n"

def translate_engine_output(output):
    aurora_probability = output["probability"]
    if output["aurora_decision"] == 1:
        message = AURORA_DETECTION_VERDICT[0] + "\n"
        message += AURORA_EXISTS
        message += f"Probability {aurora_probability}"
        return message
    else:
        if output["aurora_decision"] == -1:
            message = AURORA_DETECTION_VERDICT[2] + "\n"
            message += AURORA_UNCERTAIN + "\n"
            message += f"Probability {aurora_probability}"
        else:
            message = AURORA_DETECTION_VERDICT[1] + "\n"
            message += NO_AURORA_EXISTS + "\n"
            message += f"Probability {aurora_probability}"

        if len(output["secondary_phenomena"]) == 0:
            message += NO_AURORALIKE_PHENOMENA_EXISTS
            return message
        else:
            message += AURORALIKE_PHENOMENA_EXISTS + "\n"
            for result_item in output["secondary_phenomena"].items():
                message += f"{result_item[0]}: {result_item[1]}\n"
    log(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    log(message)
    return message