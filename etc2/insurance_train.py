import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# 모델과 데이터셋 임포트
from models import CRNN  # 기존 모델 구조 사용
from insurance_dataset import InsuranceFormDataset, collate_fn

# 경로 설정
image_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\images"
annotation_folder = r"C:\Users\user\OneDrive\Desktop\Korean_model\annotations"
save_dir = r"C:\Users\user\OneDrive\Desktop\Korean_model\insurance_models"
os.makedirs(save_dir, exist_ok=True)

# 하이퍼파라미터
batch_size = 32
learning_rate = 0.00005  # 학습률 낮춤
num_epochs = 30
early_stopping_patience = 5

# 데이터셋 및 데이터로더 생성
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = InsuranceFormDataset(image_folder, annotation_folder, transform=transform)

# 학습/검증 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 모델 초기화
output_size = len(dataset.char_to_idx)  # 문자 집합 크기
hidden_size = 256

model = CRNN(num_chars=output_size, hidden_size=hidden_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CTCLoss(blank=0, reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # AdamW 사용 및 가중치 감쇠 추가
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

# 학습 함수
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, targets, target_lengths in tqdm(loader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        
        # 모델 출력
        outputs = model(images)
        
        # 출력 크기 조정 (time_steps, batch_size, num_classes)
        outputs = outputs.permute(1, 0, 2)
        
        # 입력 길이 계산
        input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
        
        # CTC 손실 계산
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        
        loss.backward()
        # 그래디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# 검증 함수
def validate(model, loader, criterion, device, idx_to_char):
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = 0
    
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(loader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # 모델 출력
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            
            # 입력 길이 계산
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
            
            # 손실 계산
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # 예측 디코딩
            _, predictions = outputs.max(2)
            predictions = predictions.transpose(0, 1).detach().cpu().numpy()
            
            # 정확도 계산
            target_start_idx = 0
            for i, target_length in enumerate(target_lengths):
                target_text = ''.join([idx_to_char[t.item()] for t in targets[target_start_idx:target_start_idx+target_length]])
                target_start_idx += target_length
                
                # CTC 디코딩 (중복 및 blank 제거)
                pred = predictions[i]
                decoded_pred = []
                prev = -1
                for p in pred:
                    if p != 0 and p != prev:  # 0은 blank, 중복 제거
                        decoded_pred.append(p)
                    prev = p
                
                pred_text = ''.join([idx_to_char.get(p, '') for p in decoded_pred])
                
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
    
    return total_loss / len(loader), char_accuracy, word_accuracy

# 학습 실행
train_losses = []
val_losses = []
char_accuracies = []
word_accuracies = []

best_val_loss = float('inf')
early_stopping_counter = 0

# 학습 시작 전 모델 상태 저장
torch.save({
    'epoch': 0,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'char_to_idx': dataset.char_to_idx,
    'idx_to_char': dataset.idx_to_char
}, os.path.join(save_dir, 'initial_model.pth'))

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, char_accuracy, word_accuracy = validate(model, val_loader, criterion, device, dataset.idx_to_char)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    char_accuracies.append(char_accuracy)
    word_accuracies.append(word_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Char Accuracy: {char_accuracy:.4f}, Word Accuracy: {word_accuracy:.4f}")
    
    # 학습률 조정
    scheduler.step(val_loss)
    
    # 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'char_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'char_to_idx': dataset.char_to_idx,
            'idx_to_char': dataset.idx_to_char
        }, os.path.join(save_dir, 'best_insurance_ocr_model.pth'))
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
    
    # 매 에폭마다 체크포인트 저장
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'char_accuracy': char_accuracy,
        'word_accuracy': word_accuracy
    }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

# 학습 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(char_accuracies, label='Char Accuracy')
plt.plot(word_accuracies, label='Word Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_results.png'))
plt.show()

print("학습 완료!")
