from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator
import numpy as np
import cv2
from dotenv import load_dotenv
import torch
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './models/best.pt'
MODEL = YOLO(MODEL_PATH, verbose=False)

names_dict = {
    0: 'plane',
    1: 'helicopter'
}


def preprocessing(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (512, 512))
    return image


def postproccessing(img: np.ndarray, result: Results) -> np.ndarray:
    annotator = Annotator(img)
    for bbox in result.boxes:
        box = bbox.xyxy.cpu()[0]
        cls = int(bbox.cls.cpu().tolist()[0])
        cls = names_dict[cls]
        annotator.box_label(box, label=cls)
    return img


def predict(image: np.ndarray) -> np.ndarray:
    image = preprocessing(image)
    result = MODEL(image, verbose=False)[0]
    image = postproccessing(image, result)
    return image
