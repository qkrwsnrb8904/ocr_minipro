import os
import random
import shutil
from tqdm import tqdm

# 원본 및 대상 디렉토리 설정
src_img_dir = "C:/Users/user/OneDrive/Desktop/053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/원천데이터/TS1/HW-OCR/4.Validation/P.Paper/O.Form"
src_label_dir = "C:/Users/user/OneDrive/Desktop/053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/라벨링데이터/TL/라벨/HW-OCR/4.Validation/P.Paper/O.Form"

# 샘플링된 데이터를 저장할 디렉토리
sample_dir = "C:/Users/user/OneDrive/Desktop/korean_model/OCR_sample_data"
sample_img_dir = os.path.join(sample_dir, "images")
sample_label_dir = os.path.join(sample_dir, "labels")

# 디렉토리 생성
os.makedirs(sample_img_dir, exist_ok=True)
os.makedirs(sample_label_dir, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(src_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 샘플링할 파일 수 (예: 전체 데이터의 10%)
sample_size = min(500, len(image_files))
sampled_files = random.sample(image_files, sample_size)

print(f"전체 이미지: {len(image_files)}, 샘플링: {sample_size}")

# 선택된 파일 복사
for img_file in tqdm(sampled_files, desc="데이터 샘플링"):
    # 이미지 파일 복사
    shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(sample_img_dir, img_file))
    
    # 관련 JSON 파일 복사
    json_file = os.path.splitext(img_file)[0] + '.json'
    if os.path.exists(os.path.join(src_label_dir, json_file)):
        shutil.copy(os.path.join(src_label_dir, json_file), os.path.join(sample_label_dir, json_file))
    else:
        print(f"경고: {json_file} 라벨 파일을 찾을 수 없습니다.")

print(f"샘플링 완료: {sample_dir} 디렉토리에 {sample_size}개의 샘플이 저장되었습니다.")
