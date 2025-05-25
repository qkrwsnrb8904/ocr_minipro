import easyocr
import cv2
import matplotlib.pyplot as plt
import os
import json
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# EasyOCR 리더 초기화
reader = easyocr.Reader(['ko', 'en'])

# 이미지와 JSON 파일 경로
image_folder = 'C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\보험_확인서\\원천'
json_folder = 'C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\보험_확인서\\라벨'

# 이미지 파일 목록
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

if not image_files:
    print(f"오류: 지정된 폴더에 이미지 파일이 없습니다: {image_folder}")
else:
    # 첫 번째 이미지 파일 사용
    image_file = image_files[0]
    image_path = os.path.join(image_folder, image_file)
    print(f"이미지 파일 경로: {image_path}")
    
    # 이미지 파일명에서 식별자 추출
    image_identifier = os.path.splitext(image_file)[0]
    
    # 해당 이미지의 JSON 파일 찾기
    json_file = f"{image_identifier}.json"
    json_path = os.path.join(json_folder, json_file)
    
    if not os.path.exists(json_path):
        print(f"오류: JSON 파일을 찾을 수 없습니다: {json_path}")
    else:
        print(f"JSON 파일 경로: {json_path}")
        
        # JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # JSON 파일에서 bbox 데이터 추출
        json_bbox_data = json_data.get('bbox', [])
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"오류: 이미지를 불러올 수 없습니다: {image_path}")
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 결과 시각화
            plt.figure(figsize=(12, 12))
            plt.imshow(image_rgb)
            
            # JSON 바운딩 박스 위치에서 텍스트 인식
            json_results = []
            
            for json_item in json_bbox_data:
                json_id = json_item.get('id')
                json_text = json_item.get('data', '')
                x_coords = json_item.get('x', [])
                y_coords = json_item.get('y', [])
                
                if not x_coords or not y_coords:
                    continue
                
                # 바운딩 박스 좌표 계산
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 바운딩 박스 영역 추출
                bbox_region = image_rgb[y_min:y_max, x_min:x_max]
                
                if bbox_region.size == 0:
                    continue
                
                # EasyOCR로 해당 영역 텍스트 인식
                try:
                    ocr_results = reader.readtext(bbox_region)
                    ocr_text = ' '.join([text for _, text, _ in ocr_results]) if ocr_results else ''
                except Exception as e:
                    print(f"OCR 오류 (ID {json_id}): {str(e)}")
                    ocr_text = ''
                
                # 결과 저장
                json_results.append({
                    'id': json_id,
                    'json_text': json_text,
                    'ocr_text': ocr_text,
                    'x': [x_min, x_max, x_max, x_min],
                    'y': [y_min, y_min, y_max, y_max]
                })
                
                # 사각형 그리기
                plt.plot([x_min, x_max, x_max, x_min, x_min], 
                         [y_min, y_min, y_max, y_max, y_min], 'r-')
                
                # ID와 텍스트 표시
                plt.text(x_min, y_min - 10, f"ID {json_id}: JSON={json_text}, OCR={ocr_text}", 
                        color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            plt.title("JSON 바운딩 박스 위치의 EasyOCR 인식 결과")
            plt.axis('off')
            plt.show()
            
            # 결과 출력
            print("\n=== JSON ID별 텍스트 인식 결과 ===")
            for item in json_results:
                print(f"ID {item['id']}: JSON={item['json_text']}, EasyOCR={item['ocr_text']}")
