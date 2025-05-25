# json_check.py
import os
import json

annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\annotation"

# 폴더 존재 확인
if not os.path.exists(annotation_folder):
    print(f"어노테이션 폴더가 존재하지 않습니다: {annotation_folder}")
    exit()

# 폴더 내 파일 목록 확인
files = os.listdir(annotation_folder)
print(f"어노테이션 폴더 내 파일 수: {len(files)}")
print(f"파일 목록: {files[:5] if len(files) > 5 else files}")  # 처음 5개 파일만 출력

# 첫 번째 JSON 파일 구조 확인
for json_file in files:
    if json_file.endswith('.json'):
        json_path = os.path.join(annotation_folder, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n파일: {json_file}")
            print(f"최상위 키: {list(data.keys())}")
            
            # 몇 가지 일반적인 키 확인
            for key in ['annotations', 'images', 'info', 'text']:
                if key in data:
                    print(f"'{key}' 키 발견: {type(data[key])}")
                    if isinstance(data[key], list) and len(data[key]) > 0:
                        print(f"첫 번째 항목 키: {list(data[key][0].keys())}")
            
            # 전체 구조 출력 (간략화)
            print("\nJSON 구조 요약:")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:500] + "...")  # 처음 500자만 출력
            
            break  # 첫 번째 파일만 확인
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {json_file}, 오류: {str(e)}")
