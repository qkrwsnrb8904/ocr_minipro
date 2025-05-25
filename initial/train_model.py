import os
import sys
sys.path.append('C:/Users/user/OneDrive/Desktop/OCR_mini/EasyOCR')
from trainer.train import train

# 현재 작업 디렉토리 확인
print(f"현재 작업 디렉토리: {os.getcwd()}")

# 학습 실행
train(
    opt_yaml='./user_network_dir/custom.yaml',
    train_data='./training',
    valid_data='./validation',
    output_dir='./user_network_dir',
    batch_size=32,
    num_epochs=100,
    num_workers=4,
    valInterval=2000,
    saved_model='./pre_trained_model/korean_g2.pth',
    FT=True,
    lr=0.0001
)

print("모델 학습이 완료되었습니다.")
