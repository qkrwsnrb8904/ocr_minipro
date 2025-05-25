import json
import os
import cv2
import matplotlib.pyplot as plt
from glob import glob

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def find_image_file(base_folder, identifier):
    # 가능한 이미지 확장자
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # 가능한 이미지 경로 패턴
    possible_paths = []
    
    # 기본 경로
    for ext in extensions:
        possible_paths.append(os.path.join(base_folder, f"{identifier}{ext}"))
    
    # 원천 폴더 경로
    for ext in extensions:
        possible_paths.append(os.path.join(base_folder, "원천", f"{identifier}{ext}"))
        possible_paths.append(os.path.join(base_folder, "원천데이터", f"{identifier}{ext}"))
        possible_paths.append(os.path.join(base_folder, "원천_데이터", f"{identifier}{ext}"))
        possible_paths.append(os.path.join(base_folder, "원본", f"{identifier}{ext}"))
        possible_paths.append(os.path.join(base_folder, "원본데이터", f"{identifier}{ext}"))
    
    # 상위 폴더의 원천 폴더 경로
    parent_folder = os.path.dirname(base_folder)
    for ext in extensions:
        possible_paths.append(os.path.join(parent_folder, f"{identifier}{ext}"))
        possible_paths.append(os.path.join(parent_folder, "원천", f"{identifier}{ext}"))
        possible_paths.append(os.path.join(parent_folder, "원천데이터", f"{identifier}{ext}"))
        possible_paths.append(os.path.join(parent_folder, "보험_확인서_원천", f"{identifier}{ext}"))
        possible_paths.append(os.path.join(parent_folder, "보험확인서_원천", f"{identifier}{ext}"))
    
    # 이미지 파일 찾기
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 전체 폴더에서 파일명으로 검색
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if any(file.lower() == f"{identifier.lower()}{ext}" for ext in extensions):
                return os.path.join(root, file)
    
    # 상위 폴더에서도 검색
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if any(file.lower() == f"{identifier.lower()}{ext}" for ext in extensions):
                return os.path.join(root, file)
    
    return None

def visualize_all_bbox_ids(base_folder):
    # 폴더 구조 출력
    print(f"폴더 구조 확인:")
    for root, dirs, files in os.walk(base_folder):
        level = root.replace(base_folder, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:  # 처음 5개 파일만 표시
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... 외 {len(files)-5}개 파일")
    
    # JSON 파일 목록 가져오기
    # JSON 파일 목록 가져오기
    json_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.json'):
                # conda-meta나 .vscode 폴더의 파일은 제외
                if 'conda-meta' not in root and '.vscode' not in root:
                    json_files.append(os.path.join(root, file))
    
    if not json_files:
        print(f"경로에서 JSON 파일을 찾을 수 없습니다: {base_folder}")
        return
    
    print(f"\n총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # 이미지를 찾은 파일과 찾지 못한 파일 카운트
    found_count = 0
    not_found_count = 0
    
    for json_file in json_files:
        try:
            # JSON 파일 로드
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이미지 식별자 추출
            try:
                image_identifier = data['Images']['identifier']
            except KeyError:
                # 다른 가능한 키 이름 시도
                try:
                    image_identifier = data.get('image', {}).get('identifier') or data.get('identifier')
                except:
                    # 파일 이름에서 식별자 추출
                    image_identifier = os.path.splitext(os.path.basename(json_file))[0]
            
            # 이미지 파일 찾기
            image_path = find_image_file(base_folder, image_identifier)
            
            if image_path:
                found_count += 1
                print(f"\n처리 중 ({found_count}): {os.path.basename(json_file)} -> {os.path.basename(image_path)}")
                
                # 이미지 로드
                image = cv2.imread(image_path)
                if image is None:
                    print(f"이미지를 로드할 수 없습니다: {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # bbox 데이터 추출
                try:
                    bbox_data = data['bbox']
                except KeyError:
                    print(f"JSON 파일에 bbox 데이터가 없습니다: {json_file}")
                    continue
                
                # 이미지에 바운딩 박스와 ID 표시
                plt.figure(figsize=(15, 15))
                plt.imshow(image)
                
                for box in bbox_data:
                    try:
                        box_id = box['id']
                        text_data = box.get('data', '')
                        x_coords = box['x']
                        y_coords = box['y']
                        
                        # 사각형 그리기
                        x_min, y_min = min(x_coords), min(y_coords)
                        x_max, y_max = max(x_coords), max(y_coords)
                        
                        plt.plot([x_min, x_max, x_max, x_min, x_min], 
                                [y_min, y_min, y_max, y_max, y_min], 'r-')
                        
                        # ID와 텍스트 표시
                        plt.text(x_min, y_min - 10, f"ID {box_id}: {text_data}", 
                                color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
                    except Exception as e:
                        print(f"박스 처리 중 오류: {str(e)}")
                
                plt.title(f"이미지 내 텍스트 영역과 ID - {os.path.basename(image_path)}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            else:
                not_found_count += 1
                if not_found_count <= 10:  # 처음 10개만 상세 출력
                    print(f"이미지를 찾을 수 없습니다: {image_identifier}")
                elif not_found_count == 11:
                    print("... 이하 생략 ...")
        
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {json_file}")
            print(f"오류 내용: {str(e)}")
    
    print(f"\n처리 완료: 총 {len(json_files)}개 중 {found_count}개 이미지 찾음, {not_found_count}개 이미지 찾지 못함")

# 실행
base_folder = r"C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini"
visualize_all_bbox_ids(base_folder)
