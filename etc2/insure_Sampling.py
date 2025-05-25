import os
import json
import cv2
import numpy as np
from PIL import Image

# AI Hub 데이터셋 경로 설정
image_folder = r"C:\Users\user\OneDrive\Desktop\01-1.정식개방데이터\Training\01.원천데이터\TS_금융_2.보험_2-1.신청서"
annotation_folder = r"C:\Users\user\OneDrive\Desktop\01-1.정식개방데이터\Training\02.라벨링데이터\TL_금융_2.보험_2-1.신청서"

# 데이터셋 로드 함수
def load_dataset(image_folder, annotation_folder):
    data = []
    for json_file in os.listdir(annotation_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(annotation_folder, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # 이미지 파일명 추출 (AI Hub 데이터셋 구조에 맞게 수정)
            # JSON 파일에서 이미지 파일명 정보 가져오기
            image_filename = annotation.get('images')[0].get('file_name') if 'images' in annotation and len(annotation['images']) > 0 else None
            
            if not image_filename:
                # 이미지 파일명이 없으면 JSON 파일명으로 대체
                image_filename = json_file.replace('.json', '.jpg')
            
            image_path = os.path.join(image_folder, image_filename)
            
            if os.path.exists(image_path):
                # 텍스트 영역 정보 추출 (AI Hub 데이터셋 구조에 맞게 수정)
                text_regions = []
                
                # AI Hub 보험 신청서 데이터셋의 실제 구조에 맞게 수정
                if 'annotations' in annotation:
                    for item in annotation['annotations']:
                        if 'text' in item and 'points' in item:
                            # 직접 text와 points가 있는 경우
                            text = item['text']
                            points = item['points']
                            text_regions.append({
                                'text': text,
                                'points': points
                            })
                        elif 'bbox' in item:
                            # bbox 배열이 있는 경우
                            for bbox in item['bbox']:
                                if 'text' in bbox and 'points' in bbox:
                                    text = bbox['text']
                                    points = bbox['points']
                                    text_regions.append({
                                        'text': text,
                                        'points': points
                                    })
                
                if text_regions:  # 텍스트 영역이 하나 이상 있는 경우만 추가
                    data.append({
                        'image_path': image_path,
                        'text_regions': text_regions
                    })
    
    return data

# 데이터셋 로드
dataset = load_dataset(image_folder, annotation_folder)
print(f"로드된 데이터 수: {len(dataset)}")

# 데이터 확인 (첫 번째 항목)
if len(dataset) > 0:
    print(f"첫 번째 이미지 경로: {dataset[0]['image_path']}")
    print(f"텍스트 영역 수: {len(dataset[0]['text_regions'])}")
    if len(dataset[0]['text_regions']) > 0:
        print(f"첫 번째 텍스트: {dataset[0]['text_regions'][0]['text']}")
else:
    print("데이터셋이 비어 있습니다.")
