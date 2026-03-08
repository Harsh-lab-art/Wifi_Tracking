import os
SCAN_INTERVAL_SEC = 0.5
COLLECTION_DURATION = 30
WINDOW_SIZE = 20
WINDOW_STEP = 10
LABEL_NO_PERSON = 0
LABEL_PERSON = 1
LABEL_NAMES = {0: "No Person", 1: "Person Present"}
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")
MODEL_PATH  = os.path.join(MODEL_DIR, "presence_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5