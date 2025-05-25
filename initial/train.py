import os
import json
import shutil
import cv2
import numpy as np

# 경로 설정
base_path = r"C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini"
src_img_path = os.path.join(base_path, "보험_확인서", "원천")
src_label_path = os.path.join(base_path, "보험_확인서", "라벨")
dst_train_path = os.path.join(base_path, "easyocr_dataset", "train")
dst_val_path = os.path.join(base_path, "easyocr_dataset", "val")

# 디렉토리 생성
os.makedirs(dst_train_path, exist_ok=True)
os.makedirs(dst_val_path, exist_ok=True)

# 학습/검증 데이터 분할 비율
train_ratio = 0.8

# 라벨 파일 처리 및 이미지 복사
train_gt = open(os.path.join(dst_train_path, "gt.txt"), "w", encoding="utf-8")
val_gt = open(os.path.join(dst_val_path, "gt.txt"), "w", encoding="utf-8")

# 파일 목록 가져오기
img_files = [f for f in os.listdir(src_img_path) if f.endswith('.png') or f.endswith('.jpg')]
total_files = len(img_files)
train_count = int(total_files * train_ratio)

for i, img_file in enumerate(img_files):
    # 이미지 파일 경로
    img_path = os.path.join(src_img_path, img_file)
    
    # 해당 JSON 파일 찾기
    json_file = os.path.splitext(img_file)[0] + ".json"
    json_path = os.path.join(src_label_path, json_file)
    
    if not os.path.exists(json_path):
        print(f"라벨 파일을 찾을 수 없음: {json_file}")
        continue
    
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    # 필기체 텍스트 추출 (bbox의 data 필드)
    handwritten_texts = []
    for box in label_data.get('bbox', []):
        # 필기체 텍스트 식별 로직 (예: data_type이 0인 경우 필기체로 가정)
        if box.get('data_type') == 0:  # 필기체 타입에 맞게 조정 필요
            handwritten_texts.append(box.get('data', ''))
    
    # 모든 필기체 텍스트를 하나의 문자열로 결합
    text = ' '.join(handwritten_texts)
    
    if not text:  # 필기체 텍스트가 없으면 건너뛰기
        continue
    
    # 학습/검증 데이터 분할
    if i < train_count:
        dst_path = dst_train_path
        gt_file = train_gt
    else:
        dst_path = dst_val_path
        gt_file = val_gt
    
    # 이미지 파일 복사
    shutil.copy(img_path, os.path.join(dst_path, img_file))
    
    # gt.txt 파일에 라벨 정보 추가
    gt_file.write(f"{img_file}\t{text}\n")

train_gt.close()
val_gt.close()

print(f"데이터 준비 완료: 학습 데이터 {train_count}개, 검증 데이터 {total_files - train_count}개")
