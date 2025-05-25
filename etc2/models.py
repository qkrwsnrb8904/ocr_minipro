# models.py
import torch
import torch.nn as nn
import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN 특징 추출기 (ResNet18 기반)
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # 특징 맵을 시퀀스로 변환
        self.map_to_seq = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 25))  # 여기서 25는 타겟 시퀀스 길이와 일치
        )
        
        # 양방향 LSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_size * 2, num_chars)
    
    def forward(self, x):
        # CNN을 통한 특징 추출
        features = self.cnn(x)  # [batch_size, channels, height, width]
        
        # 특징 맵을 시퀀스로 변환
        features = self.map_to_seq(features)  # [batch_size, channels, 1, width]
        features = features.squeeze(2)  # [batch_size, channels, width]
        features = features.permute(0, 2, 1)  # [batch_size, width, channels]
        
        # RNN 처리
        outputs, _ = self.rnn(features)  # [batch_size, width, hidden_size*2]
        
        # 각 시점에 대한 문자 예측
        logits = self.fc(outputs)  # [batch_size, width, num_chars]
        
        return logits
