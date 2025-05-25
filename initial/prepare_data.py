# prepare_data.py
import os
import json
import shutil

# 경로 설정
src_img_dir = "보험_확인서/원천"
src_label_dir = "보험_확인서/라벨"
train_dir = "training"
valid_dir = "validation"

# 문자 집합 수집
character_set = set()

# 데이터 분할 (8:2)
json_files = [f for f in os.listdir(src_label_dir) if f.endswith('.json')]
train_count = int(len(json_files) * 0.8)
train_files = json_files[:train_count]
valid_files = json_files[train_count:]

# 학습 데이터 준비
def prepare_data(files, target_dir, gt_file_path):
    with open(gt_file_path, 'w', encoding='utf-8') as gt_file:
        for json_file in files:
            with open(os.path.join(src_label_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 구조에 맞게 수정 필요
            # 예시: COCO 형식
            if 'images' in data and 'annotations' in data:
                for image in data['images']:
                    img_file = image['file_name']
                    img_id = image['id']
                    
                    for ann in data['annotations']:
                        if ann['image_id'] == img_id:
                            text = ann['text']
                            character_set.update(text)
                            gt_file.write(f"{img_file}\t{text}\n")
                            
                            # 이미지 복사
                            src_path = os.path.join(src_img_dir, img_file)
                            dst_path = os.path.join(target_dir, img_file)
                            if os.path.exists(src_path):
                                shutil.copy(src_path, dst_path)
            
            # 단순 리스트 형식
            elif isinstance(data, list):
                for item in data:
                    img_file = item.get('file_name', '')
                    text = item.get('text', '')
                    if img_file and text:
                        character_set.update(text)
                        gt_file.write(f"{img_file}\t{text}\n")
                        
                        # 이미지 복사
                        src_path = os.path.join(src_img_dir, img_file)
                        dst_path = os.path.join(target_dir, img_file)
                        if os.path.exists(src_path):
                            shutil.copy(src_path, dst_path)

# 데이터 준비 실행
prepare_data(train_files, train_dir, os.path.join(train_dir, "gt.txt"))
prepare_data(valid_files, valid_dir, os.path.join(valid_dir, "gt.txt"))

# 문자 집합 저장
with open('character_set.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(sorted(character_set)))

print(f"총 {len(character_set)}개의 문자가 발견되었습니다.")
print(f"학습 데이터: {len(train_files)}개, 검증 데이터: {len(valid_files)}개")
