import os
import torch
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms
from models import CRNN

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def recognize_insurance_form(model_path, image_path):
    """
    보험 신청서 이미지에서 텍스트를 인식합니다.
    """
    # 모델 로드
    checkpoint = torch.load(model_path)
    idx_to_char = checkpoint['idx_to_char']
    
    num_chars = len(idx_to_char)
    model = CRNN(num_chars=num_chars)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # GPU 사용 가능 시 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 이미지 전처리 함수
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    # 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 적응형 이진화 적용 (인식률 향상)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 노이즈 제거
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 텍스트 영역 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 결과 이미지
    result_image = image.copy()
    
    # 인식 결과
    recognized_texts = []
    
    # 컨투어 정렬 (위에서 아래로, 왼쪽에서 오른쪽으로)
    def sort_contours(cnts):
        # y 좌표 기준으로 그룹화 (같은 줄에 있는 텍스트)
        y_tolerance = 20  # 같은 줄로 간주할 y 좌표 차이
        lines = {}
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            found = False
            
            for line_y in lines.keys():
                if abs(y - line_y) < y_tolerance:
                    lines[line_y].append((x, y, w, h, c))
                    found = True
                    break
            
            if not found:
                lines[y] = [(x, y, w, h, c)]
        
        # 각 줄 내에서 x 좌표로 정렬
        sorted_lines = []
        for y in sorted(lines.keys()):
            sorted_lines.append(sorted(lines[y], key=lambda x: x[0]))
        
        # 모든 컨투어를 하나의 리스트로 병합
        sorted_contours = []
        for line in sorted_lines:
            sorted_contours.extend([item[4] for item in line])
        
        return sorted_contours
    
    # 컨투어 정렬
    sorted_contours = sort_contours(contours)
    
    # 각 텍스트 영역에 대해 처리
    for i, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # 너무 작은 영역은 무시
        if w < 10 or h < 10:
            continue
        
        # 너무 큰 영역도 무시
        if w > image.shape[1] * 0.9 or h > image.shape[0] * 0.1:
            continue
        
        # 종횡비가 너무 극단적인 경우 무시
        aspect_ratio = w / float(h)
        if aspect_ratio > 20 or aspect_ratio < 0.1:
            continue
        
        # 텍스트 영역 추출
        roi = binary[y:y+h, x:x+w]
        
        # 패딩 추가 (인식률 향상)
        pad = 5
        roi_padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        # 중요: 흑백 이미지를 3채널 RGB로 변환
        roi_rgb = cv2.cvtColor(roi_padded, cv2.COLOR_GRAY2RGB)
        
        # PIL 이미지로 변환
        roi_pil = Image.fromarray(roi_rgb)
        
        # 전처리 및 모델 입력
        input_tensor = transform(roi_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            outputs = outputs.permute(1, 0, 2)
            
            # 그리디 디코딩
            _, predicted_indices = torch.max(outputs.permute(1, 0, 2), dim=2)
            
            # CTC 디코딩 (중복 제거 및 빈칸 제거)
            pred_text = ""
            prev_idx = -1
            for idx in predicted_indices[0]:
                idx = idx.item()
                if idx != 0 and idx != prev_idx:  # 0은 빈칸(blank)
                    if idx in idx_to_char:
                        pred_text += idx_to_char[idx]
                prev_idx = idx
        
        # 빈 텍스트는 무시
        if not pred_text:
            continue
        
        # 결과 저장
        recognized_texts.append({
            'text': pred_text,
            'bbox': (x, y, w, h)
        })
        
        # 결과 이미지에 바운딩 박스 및 인식된 텍스트 표시
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 텍스트 표시 (한글 지원을 위해 PIL 사용)
        result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(result_pil)
        
        # 폰트 설정 (한글 폰트 필요)
        try:
            font = ImageFont.truetype("malgun.ttf", 15)  # Windows 기본 한글 폰트
            draw.text((x, y-20), pred_text, font=font, fill=(0, 255, 0))
        except:
            # 폰트가 없는 경우 OpenCV로 표시 (한글이 깨질 수 있음)
            result_image = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            cv2.putText(result_image, pred_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 결과 이미지 저장
    cv2.imwrite('insurance_form_result.jpg', result_image)
    
    # 결과 시각화
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('인식된 텍스트')
    plt.tight_layout()
    plt.savefig('insurance_form_visualization.jpg')
    plt.show()
    
    print(f"총 {len(recognized_texts)}개의 텍스트가 인식되었습니다.")
    
    # 인식된 텍스트 출력
    print("\n인식된 텍스트:")
    for i, text_info in enumerate(recognized_texts):
        print(f"{i+1}. {text_info['text']}")
    
    return recognized_texts, result_image

# 메인 함수
if __name__ == "__main__":
    model_path = "models/best_model.pth"
    insurance_form_path = r"C:\Users\user\OneDrive\Desktop\01-1.정식개방데이터\Training\01.원천데이터\TS_금융_2.보험_2-1.신청서\IMG_OCR_6_F_0000111.png"  # 보험 신청서 이미지 경로
      
    if not os.path.exists(insurance_form_path):
        print(f"보험 신청서 이미지를 찾을 수 없습니다: {insurance_form_path}")
    else:
        # 보험 신청서 인식
        texts, result_image = recognize_insurance_form(model_path, insurance_form_path)
