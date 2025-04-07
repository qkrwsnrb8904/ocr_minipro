import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import CRNN
from tqdm import tqdm
from dataset import InsuranceOCRDataset

def finetune_insurance_ocr():
    # 경로 설정
    model_path = "models/best_model.pth"
    ts_img_dir = r"C:\Users\user\OneDrive\Desktop\01-1.정식개방데이터\Training\01.원천데이터\TS_금융_2.보험_2-1.신청서"
    tl_label_dir = r"C:\Users\user\OneDrive\Desktop\01-1.정식개방데이터\Training\02.라벨링데이터\TL_금융_2.보험_2-1.신청서"
    vs_img_dir = r"C:\Users\user\OneDrive\Desktop\01-1.정식개방데이터\Validation\01.원천데이터\VS_금융_2.보험_2-1.신청서"
    vl_label_dir = r"C:\Users\user\OneDrive\Desktop\01-1.정식개방데이터\Validation\02.라벨링데이터\VL_금융_2.보험_2-1.신청서"
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    checkpoint = torch.load(model_path)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    
    # 모델 초기화
    num_chars = len(char_to_idx)
    model = CRNN(num_chars=num_chars)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 로드
    ts_dataset = InsuranceOCRDataset(ts_img_dir, tl_label_dir, char_to_idx=char_to_idx, transform=transform)
    vs_dataset = InsuranceOCRDataset(vs_img_dir, vl_label_dir, char_to_idx=char_to_idx, transform=transform)
    
    print(f"TS 데이터셋: {len(ts_dataset)} 샘플")
    print(f"VS 데이터셋: {len(vs_dataset)} 샘플")
    
    # 데이터 로더 생성
    batch_size = 16
    train_loader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(vs_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 파인튜닝 실행
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = batch['image'].to(device)
            targets = batch['text_indices'].to(device)
            target_lengths = batch['text_length'].to(device)
            
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # [seq_len, batch, num_classes]
            
            # 입력 시퀀스 길이 계산
            input_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)
            
            # CTC 손실 계산
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 검증
        model.eval()
        total_val_loss = 0
        correct_chars = 0
        total_chars = 0
        correct_words = 0
        total_words = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = batch['image'].to(device)
                targets = batch['text_indices'].to(device)
                target_lengths = batch['text_length'].to(device)
                
                # 순전파
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)  # [seq_len, batch, num_classes]
                
                # 입력 시퀀스 길이 계산
                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(device)
                
                # CTC 손실 계산
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                total_val_loss += loss.item()
                
                # 정확도 계산 (CTC 디코딩)
                _, predicted_indices = torch.max(outputs.permute(1, 0, 2), dim=2)
                
                for i in range(len(predicted_indices)):
                    pred_text = ""
                    true_text = batch['text'][i]
                    text_length = target_lengths[i].item()
                    
                    # 그리디 디코딩 (중복 제거 및 빈칸 제거)
                    prev_idx = -1
                    for j in range(predicted_indices.size(1)):
                        pred_idx = predicted_indices[i][j].item()
                        if pred_idx != 0 and pred_idx != prev_idx:  # 0은 빈칸(blank)
                            if pred_idx in idx_to_char:
                                pred_text += idx_to_char[pred_idx]
                        prev_idx = pred_idx
                    
                    if pred_text == true_text:
                        correct_words += 1
                    total_words += 1
                    
                    # 문자 단위 정확도
                    for a, b in zip(pred_text, true_text):
                        if a == b:
                            correct_chars += 1
                    total_chars += max(len(pred_text), len(true_text))
        
        avg_val_loss = total_val_loss / len(val_loader)
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        word_accuracy = correct_words / total_words if total_words > 0 else 0
        
        # 학습률 조정
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Char Acc: {char_accuracy:.4f}, "
              f"Word Acc: {word_accuracy:.4f}")
        
        # 모델 저장 및 얼리 스토핑 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
            }, "models/insurance_finetuned_model.pth")
            print(f"Model saved at epoch {epoch+1}")
            counter = 0  # 카운터 리셋
        else:
            counter += 1  # 개선 없음 카운터 증가
            
        # 얼리 스토핑 체크
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("파인튜닝 완료!")

if __name__ == "__main__":
    finetune_insurance_ocr()
