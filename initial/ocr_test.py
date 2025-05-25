import easyocr
import os
import json
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
label_path = r"C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\보험_확인서\\라벨"
image_path = r"C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\보험_확인서\\원천"
results_path = r"C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\결과"

# 결과 디렉토리 생성
os.makedirs(results_path, exist_ok=True)

# EasyOCR 리더 초기화
reader = easyocr.Reader(['ko'])  # 한국어 설정

def process_image(image_file):
    """이미지 파일을 처리하여 OCR 결과를 반환합니다."""
    try:
        image_file_path = os.path.join(image_path, image_file)
        image = Image.open(image_file_path)
        # PIL Image를 numpy 배열로 변환
        image_np = np.array(image)
        result = reader.readtext(image_np)
        return result
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        return []

def clean_filename(filename):
    """파일명에서 'F ' 접두사를 제거합니다."""
    if filename.startswith('F '):
        return filename[2:]
    return filename

def find_matching_image(identifier):
    """식별자와 일치하는 이미지 파일을 찾습니다."""
    for img_file in os.listdir(image_path):
        # 파일명에서 'F ' 접두사를 제거하고 비교
        clean_img_file = clean_filename(img_file)
        # 식별자가 파일명에 포함되어 있는지 확인
        if identifier in clean_img_file:
            return img_file
    return None

# 디렉토리 존재 여부 확인
if not os.path.exists(label_path):
    print(f"라벨 디렉토리가 존재하지 않습니다: {label_path}")
    exit()

if not os.path.exists(image_path):
    print(f"이미지 디렉토리가 존재하지 않습니다: {image_path}")
    exit()

print(f"라벨 디렉토리 내 파일 목록: {os.listdir(label_path)}")
print(f"이미지 디렉토리 내 파일 목록: {os.listdir(image_path)}")

# 결과 저장을 위한 딕셔너리
all_results = {}

# JSON 파일 처리
for json_file in os.listdir(label_path):
    if json_file.endswith('.json'):
        json_file_path = os.path.join(label_path, json_file)
        print(f"\n처리 중인 JSON 파일: {json_file}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 이미지 파일명 생성
            image_identifier = label_data['Images']['identifier']
            image_type = label_data['Images']['type'].lower()
            image_file = f"{image_identifier}.{image_type}"
            image_file_path = os.path.join(image_path, image_file)
            
            print(f"찾는 이미지 파일: {image_file}")
            
            # 이미지 파일 존재 여부 확인
            if os.path.exists(image_file_path):
                print(f"이미지 파일을 찾았습니다: {image_file}")
                ocr_result = process_image(image_file)
                used_image = image_file
            else:
                # 이미지 파일이 없는 경우 대체 이미지 찾기
                print(f"파일이 존재하지 않습니다: {image_file_path}")
                print(f"다른 이미지 파일 찾는 중...")
                
                # 파일명에 identifier가 포함된 이미지 찾기
                matching_image = find_matching_image(image_identifier)
                
                if matching_image:
                    print(f"대체 이미지를 찾았습니다: {matching_image}")
                    ocr_result = process_image(matching_image)
                    used_image = matching_image
                else:
                    print(f"JSON 파일: {json_file}에서 참조하는 이미지를 찾을 수 없습니다.")
                    continue
            
            # OCR 결과와 라벨 데이터 비교 및 처리
            match_count = 0
            match_results = []
            
            for bbox in label_data['bbox']:
                bbox_text = bbox['data']
                bbox_id = bbox.get('id', 'unknown')
                found = False
                best_match = ""
                
                for ocr_box in ocr_result:
                    ocr_text = ocr_box[1]  # OCR 결과 텍스트
                    
                    if bbox_text in ocr_text or ocr_text in bbox_text:
                        print(f"일치: ID {bbox_id} - {bbox_text}")
                        match_count += 1
                        found = True
                        best_match = ocr_text
                        break
                
                if not found:
                    print(f"불일치: ID {bbox_id} - {bbox_text}")
                
                # 결과 저장
                match_results.append({
                    "id": bbox_id,
                    "label_text": bbox_text,
                    "matched": found,
                    "ocr_text": best_match if found else ""
                })
            
            print(f"총 {len(label_data['bbox'])}개 중 {match_count}개 일치")
            
            # 결과 저장
            all_results[json_file] = {
                "image_file": used_image,
                "total_labels": len(label_data['bbox']),
                "matched_count": match_count,
                "match_rate": round(match_count / len(label_data['bbox']) * 100, 2) if len(label_data['bbox']) > 0 else 0,
                "details": match_results
            }
        
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()

# 결과를 JSON 파일로 저장
results_file = os.path.join(results_path, "ocr_results.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n결과가 저장되었습니다: {results_file}")

# 요약 결과 출력
print("\n===== OCR 결과 요약 =====")
total_labels = 0
total_matches = 0

for json_file, result in all_results.items():
    total_labels += result["total_labels"]
    total_matches += result["matched_count"]
    print(f"{json_file}: {result['matched_count']}/{result['total_labels']} 일치 ({result['match_rate']}%)")

if total_labels > 0:
    overall_match_rate = round(total_matches / total_labels * 100, 2)
    print(f"\n전체 일치율: {total_matches}/{total_labels} ({overall_match_rate}%)")
