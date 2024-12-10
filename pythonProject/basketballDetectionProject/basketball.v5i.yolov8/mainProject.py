# 내가 갖고있는 데이터셋으로 전이학습 하는 코드
import os
from ultralytics import YOLO
from multiprocessing import freeze_support

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    freeze_support()
    # Train the model
    model.train(data='D:/박효제/2-2/AImodelProjects/pythonProject/basketballDetectionProject/basketball.v5i.yolov8/data.yaml', epochs=25, imgsz=320, device=0)