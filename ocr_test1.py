import os
import json
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

model_path = r"C:\Users\user\OneDrive\Desktop\Korean_model\finetuned_models\insurance_model_epoch_21.pth"

checkpoint = torch.load(model_path, map_location=device)
char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']


from models import CRNN

output_size = checkpoint['model_state_dict']['fc.bias'].size(0)

# 모델 초기화 (체크포인트의 출력 크기 사용)
model = CRNN(num_chars=output_size, hidden_size=256)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("모델 초기화 완료")

# 문서 유형별 필드 매핑
field_mapping = {
    "간병인_신청서": [
        "이름", "주민번호", "연락처", "주소1", "주소2", "주소3", "주소4", 
        "병원명", "치료시작일", "진단명", "신청일자", "입원호실", "입원시작일", 
        "서비스신청일", "작성년", "작성월", "작성일", "서비스신청자"
    ],
    "자동이체_신청서": [
        "증권번호", "상품명", "예금주명", "예금주 생년월일", "금융기관명", 
        "출금계좌번호", "예금주연락처", "예금주/계약자관계", "신청인명", 
        "신청인연락처", "연도", "월", "일", "계약자", "예금주"
    ]
}

def preprocess_image_for_model(image):
    """이미지 전처리"""
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 전처리
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_image).unsqueeze(0)

def recognize_text_with_model(image_tensor):
    """모델을 사용하여 텍스트 인식"""
    with torch.no_grad():
        # 모델 예측
        outputs = model(image_tensor.to(device))
        outputs = outputs.permute(1, 0, 2)
        
        # 예측 디코딩
        _, predictions = outputs.max(2)
        predictions = predictions.transpose(0, 1).detach().cpu().numpy()[0]
        
        # CTC 디코딩
        decoded_text = []
        prev = -1
        for p in predictions:
            if p != 0 and p != prev:  # 0은 blank, 중복 제거
                if p in idx_to_char:
                    decoded_text.append(idx_to_char[p])
            prev = p
        
        return ''.join(decoded_text)

def extract_text_from_bbox(image, bbox):
    """바운딩 박스에서 텍스트 추출"""
    try:
        # 바운딩 박스 좌표 추출
        x_coords = bbox["x"]
        y_coords = bbox["y"]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # 이미지에서 영역 추출
        roi = image[y1:y2, x1:x2]
        
        # 이미지 전처리
        image_tensor = preprocess_image_for_model(roi)
        
        # 모델로 텍스트 인식
        recognized_text = recognize_text_with_model(image_tensor)
        
        return recognized_text, 1.0  # 신뢰도는 모델에서 제공하지 않으므로 임의의 값 사용
    except Exception as e:
        print(f"텍스트 추출 중 오류 발생: {e}")
        return "", 0.0

def determine_document_type(annotation, image_file):
    """문서 유형 판별"""
    # 파일명으로 판별 시도
    if "11" in image_file:
        return "간병인_신청서"
    elif "14" in image_file:
        return "자동이체_신청서"
    
    # 폼 타입으로 판별 시도
    form_type = annotation.get("Images", {}).get("form_type", "").lower()
    
    if "간병" in form_type:
        return "간병인_신청서"
    elif "자동이체" in form_type or "출금" in form_type:
        return "자동이체_신청서"
    else:
        # 보험 신청서는 처리하지 않음
        return None

def process_image_with_json(image_path, json_path):
    """이미지와 JSON 파일을 처리하여 필기체 인식"""
    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 문서 유형 판별
    doc_type = determine_document_type(annotation, os.path.basename(image_path))
    print(f"문서 유형: {doc_type}")
    
    # 문서 유형에 따른 필드 매핑 선택
    field_names = field_mapping.get(doc_type, [])
    
    # 결과 저장을 위한 딕셔너리
    results = {
        'file_name': os.path.basename(image_path),
        'form_type': doc_type,
        'data_captured': annotation.get('Images', {}).get('data_captured', ''),
        'fields': {}
    }
    
    # 각 바운딩 박스 처리
    for bbox in annotation.get('bbox', []):
        field_id = bbox.get('id', 0)
        original_data = bbox.get('data', '')
        
        # 필드 이름 결정
        field_name = field_names[field_id-1] if 0 < field_id <= len(field_names) else f"필드_{field_id}"
        
        # 텍스트 추출
        recognized_text, confidence = extract_text_from_bbox(image, bbox)
        
        # 필드 유형에 따른 후처리
        if "주민번호" in field_name or "생년월일" in field_name:
            # 숫자와 하이픈만 유지
            recognized_text = ''.join(c for c in recognized_text if c.isdigit() or c == '-')
        elif "연락처" in field_name:
            # 숫자와 하이픈만 유지
            recognized_text = ''.join(c for c in recognized_text if c.isdigit() or c == '-')
        elif "날짜" in field_name or "일자" in field_name or "시작일" in field_name:
            # 날짜 형식 정규화
            digits = ''.join(c for c in recognized_text if c.isdigit())
            if len(digits) == 8: 
                recognized_text = f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"

        results['fields'][field_name] = {
            'original': original_data,
            'recognized': recognized_text
        }
        
        print(f"필드: {field_name}")
        print(f"  원본: {original_data}")
        print(f"  인식: {recognized_text}")
        print("-" * 30)
        
        # 이미지에 바운딩 박스와 인식 결과 표시
        x_coords = bbox["x"]
        y_coords = bbox["y"]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, recognized_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 결과 이미지 저장
    output_image_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"결과 이미지 저장: {output_image_path}")
    
    return results

def main():

    base_dir = r"C:\Users\user\OneDrive\Desktop\Korean_model\data\val"
    image_dir = os.path.join(base_dir, "images")
    json_dir = os.path.join(base_dir, "labels")
    output_dir = os.path.join(base_dir, "results")

    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    for image_file in image_files:
        # 이미지 경로
        image_path = os.path.join(image_dir, image_file)
        
        # JSON 파일 경로
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(json_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"JSON 파일을 찾을 수 없습니다: {json_file}")
            continue
        
        print(f"처리 중: {image_file}")
        
        # 이미지 처리
        results = process_image_with_json(image_path, json_path)
        
        if results:
            # 결과 JSON 저장
            output_json_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + '_result.json')
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"결과 JSON 저장: {output_json_path}")
            print("=" * 50)

if __name__ == "__main__":
    main()
