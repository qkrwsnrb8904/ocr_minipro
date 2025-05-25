# image_check.py
import os

image_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\images"
annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\annotations"

# 폴더 존재 확인
if not os.path.exists(image_folder):
    print(f"이미지 폴더가 존재하지 않습니다: {image_folder}")
else:
    print(f"이미지 폴더 내 파일 수: {len(os.listdir(image_folder))}")
    print(f"이미지 파일 목록 (처음 5개): {os.listdir(image_folder)[:5] if len(os.listdir(image_folder)) > 5 else os.listdir(image_folder)}")

# 어노테이션 파일에 해당하는 이미지 파일 확인
for json_file in os.listdir(annotation_folder)[:5]:  # 처음 5개만 확인
    if json_file.endswith('.json'):
        # 이미지 파일명 추출 (JSON 파일명과 동일한 이름의 이미지 파일 찾기)
        image_filename = json_file.replace('.json', '.jpg')
        image_path = os.path.join(image_folder, image_filename)
        
        if os.path.exists(image_path):
            print(f"이미지 파일 존재: {image_filename}")
        else:
            print(f"이미지 파일 없음: {image_filename}")
