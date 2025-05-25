import easyocr
import cv2
import os
import json
import numpy as np
from datetime import datetime

# EasyOCR 리더 초기화
reader = easyocr.Reader(['ko', 'en'])

# 이미지와 JSON 파일 경로
image_folder = 'C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\보험_확인서\\원천'
json_folder = 'C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\보험_확인서\\라벨'

# 결과 저장 파일
result_file = f'OCR_결과_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

# 이미지 파일 목록
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

if not image_files:
    print(f"오류: 지정된 폴더에 이미지 파일이 없습니다: {image_folder}")
else:
    print(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
    
    # 결과 파일 열기
    with open(result_file, 'w', encoding='utf-8') as result_f:
        result_f.write(f"=== EasyOCR 인식 결과 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        # 각 이미지 파일 처리
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            print(f"\n처리 중 ({idx+1}/{len(image_files)}): {image_file}")
            
            # 이미지 파일명에서 식별자 추출
            image_identifier = os.path.splitext(image_file)[0]
            
            # 해당 이미지의 JSON 파일 찾기
            json_file = f"{image_identifier}.json"
            json_path = os.path.join(json_folder, json_file)
            
            if not os.path.exists(json_path):
                print(f"  오류: JSON 파일을 찾을 수 없습니다: {json_path}")
                result_f.write(f"\n\n=== 이미지: {image_file} ===\n")
                result_f.write(f"오류: JSON 파일을 찾을 수 없습니다: {json_file}\n")
                continue
            
            # JSON 파일 로드
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # JSON 파일에서 bbox 데이터 추출
                json_bbox_data = json_data.get('bbox', [])
                
                if not json_bbox_data:
                    print(f"  오류: JSON 파일에 bbox 데이터가 없습니다: {json_path}")
                    result_f.write(f"\n\n=== 이미지: {image_file} ===\n")
                    result_f.write(f"오류: JSON 파일에 bbox 데이터가 없습니다\n")
                    continue
                
                # 이미지 로드
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  오류: 이미지를 불러올 수 없습니다: {image_path}")
                    result_f.write(f"\n\n=== 이미지: {image_file} ===\n")
                    result_f.write(f"오류: 이미지를 불러올 수 없습니다\n")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 결과 파일에 이미지 정보 기록
                result_f.write(f"\n\n=== 이미지: {image_file} ===\n")
                
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
                        print(f"  OCR 오류 (ID {json_id}): {str(e)}")
                        ocr_text = f"[OCR 오류: {str(e)}]"
                    
                    # 결과 저장
                    json_results.append({
                        'id': json_id,
                        'json_text': json_text,
                        'ocr_text': ocr_text
                    })
                    
                    # 결과 파일에 기록
                    result_f.write(f"ID {json_id}: JSON={json_text}, EasyOCR={ocr_text}\n")
                
                # 콘솔에 결과 출력
                print(f"  {len(json_results)}개의 필드를 처리했습니다.")
            
            except Exception as e:
                print(f"  파일 처리 중 오류 발생: {str(e)}")
                result_f.write(f"\n\n=== 이미지: {image_file} ===\n")
                result_f.write(f"오류: 파일 처리 중 오류 발생: {str(e)}\n")
    
    print(f"\n처리 완료! 결과가 {result_file}에 저장되었습니다.")
