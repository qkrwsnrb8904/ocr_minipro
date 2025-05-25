import easyocr
import cv2
import matplotlib.pyplot as plt
import os
import json

plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# EasyOCR 리더 초기화 (영어, 한국어 인식)
reader = easyocr.Reader(['ko', 'en'])  # GPU 사용 시 gpu=True 추가

# 이미지 폴더 경로
image_folder = 'C:\\Users\\user\\OneDrive\\Desktop\\OCR_mini\\보험_확인서\\원천'

# 폴더 내 이미지 파일 찾기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

if not image_files:
    print(f"오류: 지정된 폴더에 이미지 파일이 없습니다: {image_folder}")
else:
    # 첫 번째 이미지 파일 사용
    image_path = os.path.join(image_folder, image_files[0])
    print(f"이미지 파일 경로: {image_path}")
    
    # 이미지 로드
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"오류: 이미지를 불러올 수 없습니다: {image_path}")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 텍스트 인식 실행
        results = reader.readtext(image_rgb)
        
        # 결과 시각화
        plt.figure(figsize=(12, 12))
        plt.imshow(image_rgb)
        
        bbox_data = []  # bbox_data 리스트 초기화
        
        for i, (bbox, text, prob) in enumerate(results):
            # 바운딩 박스 좌표 추출
            (top_left, top_right, bottom_right, bottom_left) = bbox
            
            # JSON 데이터에 추가
            bbox_data.append({
                'id': i+1,
                'data': text,
                'x': [p[0] for p in [top_left, top_right, bottom_right, bottom_left]],
                'y': [p[1] for p in [top_left, top_right, bottom_right, bottom_left]]
            })
            
            # 사각형 그리기
            x_coords = [p[0] for p in [top_left, top_right, bottom_right, bottom_left, top_left]]
            y_coords = [p[1] for p in [top_left, top_right, bottom_right, bottom_left, top_left]]
            plt.plot(x_coords, y_coords, 'r-')
            
            # ID와 텍스트 표시
            plt.text(top_left[0], top_left[1] - 10, f"ID {i+1}: {text}", 
                    color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title("EasyOCR 인식 결과")
        plt.axis('off')
        plt.show()
        
        # 인식 결과 출력
        for i, (bbox, text, prob) in enumerate(results):
            print(f"ID {i+1}: {text}")
        
        # JSON 데이터 생성
        json_data = {
            'Images': {
                'identifier': os.path.splitext(os.path.basename(image_path))[0]
            },
            'bbox': bbox_data
        }
        
        # JSON 파일로 저장
        json_output_path = os.path.splitext(image_path)[0] + '_easyocr.json'
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print(f"JSON 파일이 저장되었습니다: {json_output_path}")
