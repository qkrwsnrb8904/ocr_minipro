# bbox_check.py
import os
import json

annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\annotations"

# 첫 번째 JSON 파일의 bbox 구조 확인
for json_file in os.listdir(annotation_folder):
    if json_file.endswith('.json'):
        json_path = os.path.join(annotation_folder, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"파일: {json_file}")
        
        if 'bbox' in data:
            print(f"bbox 항목 수: {len(data['bbox'])}")
            if len(data['bbox']) > 0:
                print(f"첫 번째 bbox 항목 키: {list(data['bbox'][0].keys())}")
                print(f"첫 번째 bbox 항목 내용: {data['bbox'][0]}")
        else:
            print("bbox 키가 없습니다.")
        
        break  # 첫 번째 파일만 확인
