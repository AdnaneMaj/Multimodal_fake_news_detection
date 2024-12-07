from enum import Enum

class BaseEnum(Enum):

    DATA_PATH = "./data/"
    WORD2VEC_MODEL = "src/models/embeddings/GoogleNews-vectors-negative300.bin"
    YOLO_MODEL = "./src/models/yolo/yolo11n.pt"