import tensorflow as tf
import os
import cv2
import numpy as np
from glob import glob
from models import Yolov4
import gradio as gr

# Download the checkpoints
os.system('wget -P /home/xlab-app-center/ https://huggingface.co/spaces/jbraun19/Webcam-Object-Recognition-Yolo-n-Coco/resolve/main/yolov4.weights')

model = Yolov4(weight_path="yolov4.weights", class_name_path='coco_classes.txt')
def gradio_wrapper(img):
    global model
    #print(np.shape(img))
    results = model.predict(img)
    return results[0]
demo = gr.Interface(
    gradio_wrapper,
    #gr.Image(source="webcam", streaming=True, flip=True),
    gr.Image(source="webcam", streaming=True),
    "image",
    live=True
)


demo.launch()

