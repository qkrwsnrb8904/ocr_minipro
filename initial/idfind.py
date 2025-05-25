import json
import os
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

def visualize_bbox_ids(json_file_path, image_path):
    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 이미지 로드 전 경로 확인
    print(f"이미지 경로: {image_path}")
    if not os.path.exists(image_path):
        print(f"오류: 이미지 파일이 존재하지 않습니다: {image_path}")
        return
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"오류: 이미지를 불러올 수 없습니다: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
    
    # bbox 데이터 추출
    bbox_data = data['bbox']
    
    # 이미지에 바운딩 박스와 ID 표시
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    
    for box in bbox_data:
        box_id = box['id']
        text_data = box['data']
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
    
    plt.title("이미지 내 텍스트 영역과 ID")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 실행 예시
if __name__ == "__main__":
    base_path = r"C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini"
    json_file_path = os.path.join(base_path, "보험_확인서", "라벨", "IMG_OCR_6_F_0000914.json")
  
    # JSON 파일에서 이미지 식별자 추출
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_identifier = data['Images']['identifier']
    
    # 이미지 파일 경로 설정 (여러 가능한 경로 시도)
    possible_paths = [
        os.path.join(base_path, "보험_확인서_원천", f"{image_identifier}.png"),
        os.path.join(base_path, "보험_확인서", "원천", f"{image_identifier}.png"),
        os.path.join(base_path, f"{image_identifier}.png")
    ]
    
    # 존재하는 이미지 파일 경로 찾기
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path:
        print(f"이미지 파일을 찾았습니다: {image_path}")
        visualize_bbox_ids(json_file_path, image_path)
    else:
        print("오류: 이미지 파일을 찾을 수 없습니다. 다음 경로를 시도했습니다:")
        for path in possible_paths:
            print(f"- {path}")
        
        # 사용자에게 이미지 경로 직접 입력 요청
        user_path = input("이미지 파일의 전체 경로를 입력해주세요: ")
        if os.path.exists(user_path):
            visualize_bbox_ids(json_file_path, user_path)
        else:
            print(f"오류: 입력한 경로에도 이미지 파일이 존재하지 않습니다: {user_path}")
