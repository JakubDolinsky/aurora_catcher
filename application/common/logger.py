import datetime
import os

from application import config

base_dir = os.path.join(os.path.dirname(__file__))
parent_dir = os.path.dirname(base_dir)
log_dir = os.path.join(parent_dir, "log")
os.makedirs(log_dir, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"inference_log_{run_id}.txt")

def log(msg):
    if config.ALLOW_LOGGING:
        print(msg)
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")
    else:
        pass