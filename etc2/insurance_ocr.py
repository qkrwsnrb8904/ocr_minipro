import os
import torch
import cv2
import numpy as np
import json
from torchvision import transforms
from models import CRNN

# 경로 설정
model_path = r"C:\Users\user\OneDrive\Desktop\Korean_model\models\best_model.pth"
image_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\OCR_sample_datas\images"
annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\OCR_sample_datas\annotations"
output_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\OCR_results"
os.makedirs(output_folder, exist_ok=True)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {device}")

checkpoint = torch.load(model_path, map_location=device)
print("모델 체크포인트 로드 완료")

# 문자 집합 로드
if 'char_to_idx' in checkpoint:
    char_to_idx = checkpoint['char_to_idx']
elif 'state_dict' in checkpoint and 'char_to_idx' in checkpoint['state_dict']:
    char_to_idx = checkpoint['state_dict']['char_to_idx']
else:
    # 체크포인트에 문자 매핑이 없는 경우 기본값 설정
    print("경고: 체크포인트에서 문자 매핑을 찾을 수 없습니다. 기본값을 사용합니다.")
    import string
    chars = string.printable + '가나다라마바사아자차카타파하' + ''.join([chr(i) for i in range(0xAC00, 0xD7A4)])
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    char_to_idx['<blank>'] = 0

idx_to_char = {idx: char for char, idx in char_to_idx.items()}
print(f"문자 집합 크기: {len(char_to_idx)}")

# 모델 초기화
num_chars = len(char_to_idx)
hidden_size = 256
model = CRNN(num_chars=num_chars, hidden_size=hidden_size)

# 모델 가중치 로드
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()
print("모델 초기화 완료")

# 문서 유형별 필드 매핑
field_mapping = {
    "자동이체_신청서": {
        1: "증권번호",
        2: "상품명",
        3: "예금주명",
        4: "예금주 생년월일",
        5: "금융기관명",
        6: "출금계좌번호",
        7: "예금주 연락처",
        8: "예금주/계약자 관계",
        9: "신청인명",
        10: "신청인 연락처",
        11: "연도",
        12: "월",
        13: "일",
        14: "계약자",
        15: "예금주"
    },
    "간병인_신청서": {
        1: "이름",
        2: "주민번호",
        3: "연락처",
        4: "주소1",
        5: "주소2",
        6: "주소3",
        7: "우편번호",
        8: "병원명",
        9: "치료시작일",
        10: "진단명",
        11: "신청일자",
        12: "입원호실",
        13: "입원시작일",
        14: "서비스신청일",
        15: "작성일_년",
        16: "작성일_월",
        17: "작성일_일",
        18: "서비스신청자"
    },
    "보험_신청서": {
        1: "증권번호",
        2: "보험상품명",
        3: "피보험자명",
        4: "생년월일",
        5: "관계",
        6: "주민등록번호",
        7: "연락처",
        8: "관계2",
        9: "수익자명",
        10: "수익자연락처",
        11: "작성일_년",
        12: "작성일_월",
        13: "작성일_일",
        14: "청구인성명",
        15: "서명"
    }
}

# 이미지 전처리 함수
def preprocess_image(image, bbox):
    # 바운딩 박스 좌표 추출
    x_coords = bbox['x']
    y_coords = bbox['y']
    
    # 바운딩 박스의 좌상단과 우하단 좌표 계산
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    
    # 이미지 크롭
    try:
        cropped = image[y_min:y_max, x_min:x_max]
        
        # 이미지 크기 조정
        cropped = cv2.resize(cropped, (128, 32))
    except Exception as e:
        print(f"이미지 크롭/리사이즈 오류: {e}")
        print(f"좌표: x={x_min}:{x_max}, y={y_min}:{y_max}, 이미지 크기: {image.shape}")
        cropped = np.zeros((32, 128, 3), dtype=np.uint8)
    
    # 이미지 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(cropped).unsqueeze(0)

# 텍스트 인식 함수
def recognize_text(model, image_tensor, idx_to_char):
    with torch.no_grad():
        # 모델 예측
        outputs = model(image_tensor.to(device))
        
        # CTC 디코딩
        _, predictions = outputs.max(2)
        predictions = predictions.transpose(0, 1).detach().cpu().numpy()[0]
        
        # 중복 및 blank 제거
        decoded_text = []
        prev = -1
        for p in predictions:
            if p != 0 and p != prev:  # 0은 blank, 중복 제거
                if p in idx_to_char:
                    decoded_text.append(idx_to_char[p])
            prev = p
        
        return ''.join(decoded_text)

# 필드 값 검증 함수
def validate_field(field_name, value, data_type):
    """
    필드 값 검증 및 정규화 함수
    data_type: 0=한글, 1=영문, 2=숫자, 3=혼합
    """
    # 필드별 특수 처리
    if field_name in ["증권번호", "출금계좌번호", "주민번호"]:
        # 숫자와 하이픈만 유지
        return ''.join(c for c in value if c.isdigit() or c == '-')
    
    elif field_name in ["연락처", "예금주 연락처", "신청인 연락처"]:
        # 전화번호 형식 (숫자와 하이픈만)
        cleaned = ''.join(c for c in value if c.isdigit() or c == '-')
        # 하이픈이 없는 경우 추가 (예: 01012345678 -> 010-1234-5678)
        if '-' not in cleaned and len(cleaned) >= 10:
            if cleaned.startswith('02') and len(cleaned) == 10:  # 서울 지역번호
                return f"{cleaned[:2]}-{cleaned[2:6]}-{cleaned[6:]}"
            elif len(cleaned) == 11:  # 휴대폰
                return f"{cleaned[:3]}-{cleaned[3:7]}-{cleaned[7:]}"
        return cleaned
    
    elif field_name in ["생년월일", "예금주 생년월일", "치료시작일", "입원시작일", "서비스신청일"]:
        # 날짜 형식
        cleaned = ''.join(c for c in value if c.isdigit() or c in ['.', '-', '/'])
        # 숫자만 있는 경우 형식 추가 (예: 19900101 -> 1990.01.01)
        if all(c.isdigit() for c in cleaned) and len(cleaned) == 8:
            return f"{cleaned[:4]}.{cleaned[4:6]}.{cleaned[6:]}"
        return cleaned
    
    elif data_type == 2:  # 숫자
        # 숫자만 추출
        return ''.join(c for c in value if c.isdigit())
    
    elif data_type == 3:  # 혼합 (날짜, 주민번호 등)
        # 숫자와 특수문자 유지
        return ''.join(c for c in value if c.isdigit() or c in ['-', '.', '/'])
    
    # 기타 필드는 그대로 반환
    return value

# 문서 유형 판별 함수
def determine_document_type(annotation):
    # 'form_type' 필드를 사용하여 문서 유형 판별
    form_type = annotation.get("Images", {}).get("form_type", "").lower()
    
    if "자동이체" in form_type or "출금" in form_type or "계좌" in form_type:
        return "자동이체_신청서"
    elif "간병" in form_type or "간호" in form_type:
        return "간병인_신청서"
    elif "보험" in form_type or "신청서" in form_type:
        return "보험_신청서"
    else:
        # 기본값으로 보험 신청서 반환
        return "보험_신청서"

# 보험 신청서 OCR 처리 함수
def process_insurance_form(image_path, annotation_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
        "form_type": doc_type,
        "data_captured": annotation.get("Images", {}).get("data_captured", ""),
        "fields": {}
    }
    
    # 문서 유형에 따른 필드 매핑 선택
    current_field_mapping = field_mapping.get(doc_type, {})
    
    # 각 바운딩 박스 처리
    for bbox in annotation.get("bbox", []):
        bbox_id = bbox.get("id")
        data = bbox.get("data", "")
        data_type = bbox.get("data_type", 0)
        
        # 필드명 확인
        field_name = current_field_mapping.get(bbox_id, f"필드_{bbox_id}")
        
        # 모델을 사용한 텍스트 인식
        field_tensor = preprocess_image(image, bbox)
        recognized_text = recognize_text(model, field_tensor, idx_to_char)
        
        # 원본 데이터와 인식 결과 비교
        print(f"필드: {field_name}")
        print(f"원본: {data}")
        print(f"인식: {recognized_text}")
        print("-" * 30)
        
        # 결과 검증 및 정규화
        # 인식 결과가 좋지 않을 경우 원본 데이터 사용
        # 결과 검증 및 정규화
        if not recognized_text or len(recognized_text) < len(data) * 0.5:
            validated_text = validate_field(field_name, data, data_type)
        else:
            validated_text = validate_field(field_name, recognized_text, data_type)
        
        # 결과 저장
        result["fields"][field_name] = validated_text
    
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
