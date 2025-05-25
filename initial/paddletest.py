import os
import json

# 현재 디렉토리 확인
current_dir = os.getcwd()
print(f"현재 작업 디렉토리: {current_dir}")

# 경로 설정
base_dir = r'C:\Users\user\OneDrive\Desktop\Korean_model\data\train'  # 기본 폴더
label_dir = os.path.join(base_dir, 'labels')
source_dir = os.path.join(base_dir, 'images')

# 폴더 존재 확인
if not os.path.exists(base_dir):
    print(f"오류: '{base_dir}' 폴더가 존재하지 않습니다.")
    exit(1)
if not os.path.exists(label_dir):
    print(f"오류: '{label_dir}' 폴더가 존재하지 않습니다.")
    exit(1)
if not os.path.exists(source_dir):
    print(f"오류: '{source_dir}' 폴더가 존재하지 않습니다.")
    exit(1)

print(f"라벨 폴더 경로: {label_dir}")
print(f"원천 폴더 경로: {source_dir}")

# 텍스트 인식용 데이터 준비
try:
    with open('train_list.txt', 'w', encoding='utf-8') as f:
        # 라벨 폴더 내 파일 목록 가져오기
        json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        print(f"JSON 파일 수: {len(json_files)}")
        
        for json_file in json_files:
            img_name = json_file.replace('.json', '.png')
            img_path = os.path.join(source_dir, img_name)
            json_path = os.path.join(label_dir, json_file)
            
            # 이미지 파일 존재 확인
            if not os.path.exists(img_path):
                print(f"경고: 이미지 파일이 없습니다 - {img_path}")
                continue
            
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    label_data = json.load(jf)
                    
                    # bbox 배열에서 텍스트 추출
                    if 'bbox' in label_data and isinstance(label_data['bbox'], list):
                        texts = [item['data'] for item in label_data['bbox'] if 'data' in item]
                        combined_text = ' '.join(texts)
                        
                        if combined_text:
                            f.write(f"{img_path}\t{combined_text}\n")
                        else:
                            print(f"경고: 추출된 텍스트가 없습니다 - {json_file}")
                    else:
                        print(f"경고: bbox 배열이 없거나 형식이 맞지 않습니다 - {json_file}")
            
            except json.JSONDecodeError:
                print(f"오류: JSON 파일 형식이 잘못되었습니다 - {json_path}")
            except Exception as e:
                print(f"오류 발생: {json_file} - {str(e)}")
    
    print("train_list.txt 파일이 성공적으로 생성되었습니다.")
    print("파일을 확인하여 내용이 올바른지 확인하세요.")

except Exception as e:
    print(f"오류 발생: {str(e)}")
