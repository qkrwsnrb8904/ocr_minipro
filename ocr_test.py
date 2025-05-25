import os
import torch
import cv2
import json
import numpy as np
from torchvision import transforms
from models import CRNN

# 경로 설정
model_path = r"C:\Users\user\OneDrive\Desktop\Korean_model\finetuned_models\insurance_model_epoch_20.pth"
image_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\data\val\images"
annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\data\val\labels"
output_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\OCR_results"
os.makedirs(output_folder, exist_ok=True)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

checkpoint = torch.load(model_path, map_location=device)
char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']
print(f"문자 집합 크기: {len(char_to_idx)}")

# 모델 초기화
output_size = checkpoint['model_state_dict']['fc.bias'].size(0)
model = CRNN(num_chars=output_size, hidden_size=256)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("모델 초기화 완료")

def preprocess_image(image):
    # 이미지 크기 확인 및 최소 크기 보장
    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return None  # 너무 작은 이미지는 처리하지 않음
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 이진화 (적응형 임계값)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 다시 3채널로 변환
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # 크기 조정
    processed = cv2.resize(processed, (128, 32))
    
    # 텐서 변환 및 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(processed).unsqueeze(0)

# 텍스트 인식 함수
def recognize_text(model, image_tensor, idx_to_char):
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

# 후처리 함수
def postprocess_text(text, field_type=None):
    # <blank> 태그 제거
    text = text.replace('<blank>', '')
    
    # 필드 유형에 따른 후처리
    if field_type == 'number':
        # 숫자만 추출
        digits = ''.join(c for c in text if c.isdigit())
        if digits:
            return digits
        return text
    
    elif field_type == 'date':
        # 날짜 형식 정규화
        digits = ''.join(c for c in text if c.isdigit())
        if len(digits) == 8:  # YYYYMMDD
            return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
        elif len(digits) == 6:  # YYMMDD
            return f"20{digits[:2]}.{digits[2:4]}.{digits[4:6]}"
        return text
    
    elif field_type == 'phone':
        # 전화번호 형식 정규화
        digits = ''.join(c for c in text if c.isdigit())
        if len(digits) == 11:  # 휴대폰 번호
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
        elif len(digits) == 10:
            if digits.startswith('02'):  # 서울 지역번호
                return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
            else:  # 기타 지역번호
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        elif len(digits) == 9 and digits.startswith('02'):  # 서울 지역번호(8자리)
            return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
        return digits
    
    elif field_type == 'id_number':
        # 주민등록번호 형식 정규화
        digits = ''.join(c for c in text if c.isdigit() or c == '-')
        if '-' not in digits and len(digits) >= 13:
            return f"{digits[:6]}-{digits[6:13]}"
        return digits
    
    elif field_type == 'address':
        # 주소 관련 키워드 보존
        keywords = ['시', '도', '군', '구', '읍', '면', '동', '로', '길']
        for keyword in keywords:
            if keyword in text:
                return text
    
    # 기본 후처리 (공백 제거)
    return text.replace(' ', '')

# 문서 유형별 필드 매핑 (업데이트됨)
field_mapping = {
    "간병인 신청서": {
        1: {"name": "이름", "type": "text"},
        2: {"name": "주민번호", "type": "id_number"},
        3: {"name": "연락처", "type": "phone"},
        4: {"name": "주소1", "type": "text"},
        5: {"name": "주소2", "type": "text"},
        6: {"name": "주소3", "type": "text"},
        7: {"name": "주소4", "type": "text"},
        8: {"name": "병원명", "type": "text"},
        9: {"name": "치료시작일", "type": "date"},
        10: {"name": "진단명", "type": "text"},
        11: {"name": "신청일자", "type": "date"},
        12: {"name": "입원호실", "type": "text"},
        13: {"name": "입원시작일", "type": "date"},
        14: {"name": "서비스신청일", "type": "date"},
        15: {"name": "작성년", "type": "number"},
        16: {"name": "작성월", "type": "number"},
        17: {"name": "작성일", "type": "number"},
        18: {"name": "서비스신청자", "type": "text"}
    },
    "자동이체 신청서": {
        1: {"name": "증권번호", "type": "number"},
        2: {"name": "상품명", "type": "text"},
        3: {"name": "예금주명", "type": "text"},
        4: {"name": "예금주 생년월일", "type": "id_number"},
        5: {"name": "금융기관명", "type": "text"},
        6: {"name": "출금계좌번호", "type": "number"},
        7: {"name": "예금주연락처", "type": "phone"},
        8: {"name": "예금주/계약자관계", "type": "text"},
        9: {"name": "신청인명", "type": "text"},
        10: {"name": "신청인연락처", "type": "phone"},
        11: {"name": "연도", "type": "number"},
        12: {"name": "월", "type": "number"},
        13: {"name": "일", "type": "number"},
        14: {"name": "계약자", "type": "text"},
        15: {"name": "예금주", "type": "text"}
    }
}

# 문서 유형 판별 함수
def determine_document_type(annotation):
    file_identifier = annotation.get("Dataset", {}).get("identifier", "")
    
    if file_identifier.endswith("11"):
        return "간병인 신청서"  # 언더스코어 제거
    elif file_identifier.endswith("14"):
        return "자동이체 신청서"  # 언더스코어 제거
    
    # 폼 타입으로 판별 시도
    form_type = annotation.get("Images", {}).get("form_type", "").lower()
    
    if "간병" in form_type or "간호" in form_type:
        return "간병인 신청서"
    elif "자동이체" in form_type or "출금" in form_type:
        return "자동이체 신청서"
    else:
        return "보험 신청서"

# 보험 신청서 OCR 처리 함수
def process_insurance_form(image_path, annotation_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 어노테이션 로드
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
    except Exception as e:
        print(f"어노테이션 파일을 로드할 수 없습니다: {annotation_path}, 오류: {str(e)}")
        return None
    
    # 문서 유형 판별
    doc_type = determine_document_type(annotation)
    print(f"문서 유형: {doc_type}")
    
    # 결과 저장을 위한 딕셔너리
    result = {
        "file_name": os.path.basename(image_path),
        "form_type": doc_type.replace('_', ' '),  # 언더스코어를 공백으로 변경
        "data_captured": annotation.get("Images", {}).get("data_captured", ""),
        "fields": {}
    }
    
    # 문서 유형에 따른 필드 매핑 선택
    current_field_mapping = field_mapping.get(doc_type, {})
    
    # 각 바운딩 박스 처리
    for bbox in annotation.get("bbox", []):
        bbox_id = bbox.get("id")
        data = bbox.get("data", "")
        
        # 필드 정보 확인
        field_info = current_field_mapping.get(bbox_id, {"name": f"필드_{bbox_id}", "type": "text"})
        field_name = field_info["name"]
        field_type = field_info["type"]
        
        # 바운딩 박스 좌표 추출
        x_coords = bbox.get("x", [])
        y_coords = bbox.get("y", [])
        
        if not x_coords or not y_coords:
            continue
        
        # 바운딩 박스 영역 추출
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        try:
            # 영역 크롭
            roi = image[y_min:y_max, x_min:x_max]
            
            # 이미지가 너무 작은지 확인
            if roi.shape[0] < 5 or roi.shape[1] < 5:
                print(f"필드 {field_name}의 이미지가 너무 작습니다. 원본 데이터 사용.")
                result["fields"][field_name] = data.replace(' ', '')
                continue

            # 이미지 전처리
            field_tensor = preprocess_image(roi)
            
            if field_tensor is None:
                print(f"필드 {field_name}의 이미지 전처리 실패. 원본 데이터 사용.")
                result["fields"][field_name] = data.replace(' ', '')
                continue

            # 텍스트 인식
            recognized_text = recognize_text(model, field_tensor, idx_to_char)
            
            # 후처리
            processed_text = postprocess_text(recognized_text, field_type)
            
        # 인식 결과가 비어있거나 너무 짧으면 원본 데이터 사용
            if not processed_text or (len(processed_text) < len(data) * 0.5 and len(data) > 3):
                print(f"인식 결과가 불충분합니다. 원본 데이터 사용.")
                result["fields"][field_name] = data.replace(' ', '')
            else:
                # 결과 저장 (공백 제거)
                result["fields"][field_name] = processed_text.replace(' ', '')
                
        except Exception as e:
            print(f"필드 처리 중 오류: {field_name}, {str(e)}")
            # 오류 발생 시 원본 데이터 사용
            result["fields"][field_name] = data.replace(' ', '')
    
    return result

# 메인 함수
def main():
    print("보험 신청서 OCR 처리 시작...")
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"처리할 이미지 수: {len(image_files)}")
    
    # 모든 이미지 처리
    for idx, image_file in enumerate(image_files):
        # 이미지 경로
        image_path = os.path.join(image_folder, image_file)
        
        # 어노테이션 파일명 (이미지 파일명과 동일하되 확장자만 .json)
        annotation_file = image_file.replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json')
        annotation_path = os.path.join(annotation_folder, annotation_file)
        
        if not os.path.exists(annotation_path):
            print(f"어노테이션 파일이 없습니다: {annotation_file}")
            continue
        
        print(f"\n[{idx+1}/{len(image_files)}] 처리 중: {image_file}")
        result = process_insurance_form(image_path, annotation_path)
        
        if result:
            # JSON 파일로 저장
            output_path = os.path.join(output_folder, image_file.replace('.jpg', '').replace('.png', '').replace('.jpeg', '') + '_result.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            
            print(f"결과 저장: {output_path}")
    
    print("\n모든 이미지 처리 완료!")

if __name__ == "__main__":
    main()
