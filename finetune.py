import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from models import CRNN
from dataset import InsuranceFormDataset, collate_fn


model_path = r"C:\Users\user\OneDrive\Desktop\Korean_model\models\best_model.pth"
image_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\data\train\images"
annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\data\train\labels"
output_dir = r"C:\Users\user\OneDrive\Desktop\Korean_model\finetuned_models"
os.makedirs(output_dir, exist_ok=True)

# 하이퍼파라미터
batch_size = 32  
learning_rate = 0.0005  
num_epochs = 30
 
# 기존 모델 로드
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
    print("경고: 체크포인트에서 문자 매핑을 찾을 수 없습니다.")
    # 기본 문자 집합 정의
    import string
    chars = string.printable + '가나다라마바사아자차카타파하' + ''.join([chr(i) for i in range(0xAC00, 0xD7A4)])
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    char_to_idx['<blank>'] = 0

# '<blank>' 키가 없으면 추가
if '<blank>' not in char_to_idx:
    char_to_idx['<blank>'] = 0

idx_to_char = {idx: char for char, idx in char_to_idx.items()}
print(f"문자 집합 크기: {len(char_to_idx)}")

# 체크포인트에서 출력 레이어 크기 가져오기
if 'model_state_dict' in checkpoint and 'fc.bias' in checkpoint['model_state_dict']:
    output_size = checkpoint['model_state_dict']['fc.bias'].size(0)
elif 'state_dict' in checkpoint and 'fc.bias' in checkpoint['state_dict']:
    output_size = checkpoint['state_dict']['fc.bias'].size(0)
else:
    # 출력 크기를 찾을 수 없는 경우 문자 집합 크기 사용
    output_size = len(char_to_idx)

print(f"출력 레이어 크기: {output_size}")

# 모델 초기화 (체크포인트의 출력 크기 사용)
model = CRNN(num_chars=output_size, hidden_size=256)

# 모델 가중치 로드
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
print("모델 초기화 완료")

# 데이터셋 및 데이터로더 생성
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
dataset = InsuranceFormDataset(image_folder, annotation_folder, char_to_idx, transform=transform)
print(f"로드된 전체 샘플 수: {len(dataset)}")

# 데이터셋이 비어있는지 확인
if len(dataset) == 0:
    raise ValueError("데이터셋이 비어 있습니다. 경로와 파일을 확인하세요.")

# 데이터셋 테스트
try:
    print("데이터셋 테스트 중...")
    sample_image, sample_target, sample_length = dataset[0]
    print(f"샘플 이미지 크기: {sample_image.shape}")
    print(f"샘플 타겟 길이: {sample_length}")
    print("데이터셋 테스트 완료")
except Exception as e:
    print(f"데이터셋 테스트 실패: {str(e)}")
    raise

# 데이터 로더 생성
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
print(f"학습 데이터 로더 생성 완료: {len(train_loader)} 배치")

# 손실 함수 및 옵티마이저
criterion = nn.CTCLoss(blank=0, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 함수
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for images, targets, target_lengths in tqdm(loader, desc="학습 중"):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        
        # 모델 출력
        outputs = model(images)
        
        # 출력 크기 조정
        outputs = outputs.permute(1, 0, 2)
        
        # 입력 길이 계산
        input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
        
        # CTC 손실 계산
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 그래디언트 클리핑
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0

# 파인튜닝 실행
for epoch in range(num_epochs):
    print(f"\n에폭 {epoch+1}/{num_epochs}")
    
    # 학습
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"학습 손실: {train_loss:.4f}")
    
    # 매 에폭마다 체크포인트 저장
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, os.path.join(output_dir, f'insurance_model_epoch_{epoch+1}.pth'))
    print(f"모델 저장: insurance_model_epoch_{epoch+1}.pth")

# 최종 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char
}, os.path.join(output_dir, 'insurance_ocr_model.pth'))
print("\n파인튜닝 완료! 최종 모델 저장: insurance_ocr_model.pth")
