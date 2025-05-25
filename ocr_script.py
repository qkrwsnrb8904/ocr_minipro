import os
import json
from ocr_script import PaddleOCR
import cv2
import numpy as np
import matplotlib.pyplot as plt

# PaddleOCR 초기화 (한국어 모델 사용)
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 경로 설정
base_dir = r'C:\Users\user\OneDrive\Desktop\Korean_model\data\val'  # 기본 폴더
label_dir = os.path.join(base_dir, 'labels')
source_dir = os.path.join(base_dir, 'images')

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

# 각 이미지에 대해 처리
for img_file in image_files:
    img_path = os.path.join(source_dir, img_file)
    json_file = os.path.join(label_dir, img_file.replace('.png', '.json'))
    
    # 이미지 로드
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # OCR 실행
    result = ocr.ocr(img_path, cls=True)
    
    # JSON 파일이 존재하는 경우 라벨 정보 로드
    label_info = None
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            label_info = json.load(f)
    
    # 결과 시각화
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    
    # 결과 출력
    print(f"파일: {img_file}")
    for idx, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
        print(f"텍스트 {idx+1}: {txt}, 신뢰도: {score:.4f}")
    
    # 이미지에 결과 표시
    img_with_boxes = image.copy()
    for box in boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_with_boxes, [box], True, (0, 255, 0), 2)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(img_with_boxes)
    plt.title(f"OCR 결과: {img_file}")
    plt.axis('off')
    plt.show()
