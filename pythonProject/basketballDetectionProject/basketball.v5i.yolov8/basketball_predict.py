import os
from ultralytics import YOLO
from multiprocessing import freeze_support

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 모델 로드
model = YOLO('yolov8n.pt')  # 사전 학습된 모델 로드

if __name__ == '__main__':
    freeze_support()

    # 이미지를 예측하는 코드 (원래 코드)
    model.predict(source="testimage.jpg", save=True, show=True)

    # 검증 데이터셋에 대해 모델을 평가
    results = model.val(data='D:/박효제/2-2/AImodelProjects/pythonProject/basketballDetectionProject/basketball.v5i.yolov8/data.yaml')  # 검증 데이터셋의 경로 (dataset.yaml 파일)
    print(results)  # mAP, Precision, Recall 등의 지표가 출력됩니다.
