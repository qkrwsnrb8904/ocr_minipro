import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np

from models import CRNN
from dataset import InsuranceFormDataset, collate_fn

def main():
    # 경로 설정
    model_path = r"C:\Users\user\OneDrive\Desktop\Korean_model\finetuned_models\insurance_model_epoch_21.pth"
    image_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\data\val\images"
    
    annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\data\val\labels"
    
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=device)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    print(f"문자 집합 크기: {len(char_to_idx)}")
    
    # 체크포인트에서 출력 크기 가져오기
    output_size = checkpoint['model_state_dict']['fc.bias'].size(0)
    print(f"모델 출력 크기: {output_size}")
    
    # 모델 초기화 (체크포인트의 출력 크기 사용)
    model = CRNN(num_chars=output_size, hidden_size=256)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("모델 초기화 완료")
    
    # 데이터셋 및 데이터로더 생성
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 테스트 데이터셋 로드
    try:
        dataset = InsuranceFormDataset(image_folder, annotation_folder, char_to_idx, transform=transform)
        print(f"데이터셋 로드 완료: {len(dataset)} 샘플")
        
        # 테스트 샘플 수 제한
        test_size = min(500, int(len(dataset) * 0.05))  # 최대 500개 샘플 사용
        indices = torch.randperm(len(dataset))[:test_size].tolist()
        
        # 테스트 샘플 추출
        test_samples = [dataset[i] for i in indices]
        print(f"테스트 샘플 수: {len(test_samples)}")
        
        # 평가 부분 수정
        correct_chars = 0
        total_chars = 0
        correct_words = 0
        total_words = 0

        with torch.no_grad():
            for sample_idx, (image, target, target_length) in enumerate(tqdm(test_samples, desc="평가 중")):
                # 배치 차원 추가
                image = image.unsqueeze(0).to(device)
                
                # 모델 예측
                outputs = model(image)
                outputs = outputs.permute(1, 0, 2)
                
                # 예측 디코딩
                _, predictions = outputs.max(2)
                predictions = predictions.transpose(0, 1).detach().cpu().numpy()[0]
                
                # 타겟 텍스트 추출 및 <blank> 제거
                target_text = ''.join([idx_to_char[t.item()] for t in target[:target_length]])
                target_text = target_text.replace('<blank>', '')
                
                # CTC 디코딩
                decoded_pred = []
                prev = -1
                for p in predictions:
                    if p != 0 and p != prev:  # 0은 blank, 중복 제거
                        if p in idx_to_char:
                            decoded_pred.append(idx_to_char[p])
                    prev = p
                
                pred_text = ''.join(decoded_pred)
                
                # 결과 출력 (일부 샘플만)
                if sample_idx < 10 or sample_idx % 50 == 0:
                    print(f"샘플 {sample_idx}:")
                    print(f"  타겟: {target_text}")
                    print(f"  예측: {pred_text}")
                
                # 문자 정확도
                for pred_char, target_char in zip(pred_text, target_text):
                    if pred_char == target_char:
                        correct_chars += 1
                total_chars += len(target_text)
                
                # 단어 정확도
                if pred_text == target_text:
                    correct_words += 1
                total_words += 1

        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        word_accuracy = correct_words / total_words if total_words > 0 else 0

        print(f"\n문자 정확도: {char_accuracy:.4f}")
        print(f"단어 정확도: {word_accuracy:.4f}")
    
    except Exception as e:
        print(f"평가 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
