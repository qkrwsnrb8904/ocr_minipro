# dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, max_length=25):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.max_length = max_length
        self.samples = []
        
        # 문자 집합 초기화
        self.char_list = ['<pad>']  # 0번 인덱스는 패딩
        self.char_to_idx = {'<pad>': 0}
        self.idx_to_char = {0: '<pad>'}
        
        # 데이터 로드
        self._load_data()
        
        print(f"데이터셋 로드 완료: {len(self.samples)} 샘플, {len(self.char_list)} 문자")
    
    def _load_data(self):
        # 이미지 파일 목록 가져오기
        image_files = [f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 각 이미지에 대한 레이블 정보 로드
        for img_file in image_files:
            img_path = os.path.join(self.img_dir, img_file)
            label_path = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + '.json')
            
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                
                # 텍스트와 바운딩 박스 정보 추출
                for bbox_item in label_data.get('bbox', []):
                    text = bbox_item.get('data', '')
                    x_coords = bbox_item.get('x', [0, 0, 0, 0])
                    y_coords = bbox_item.get('y', [0, 0, 0, 0])
                    
                    # 바운딩 박스 형식: [x1, y1, x2, y2]
                    bbox = [
                        min(x_coords),
                        min(y_coords),
                        max(x_coords),
                        max(y_coords)
                    ]
                    
                    # 문자 집합 업데이트
                    for char in text:
                        if char not in self.char_to_idx:
                            self.char_list.append(char)
                            self.char_to_idx[char] = len(self.char_list) - 1
                            self.idx_to_char[len(self.char_list) - 1] = char
                    
                    # 샘플 추가
                    self.samples.append({
                        'image_path': img_path,
                        'text': text,
                        'bbox': bbox
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 원본 이미지 로드
        full_image = Image.open(sample['image_path']).convert('RGB')
        
        # 바운딩 박스에서 텍스트 영역 추출
        bbox = sample['bbox']
        text_region = full_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # 변환 적용
        if self.transform:
            text_region = self.transform(text_region)
        
        # 텍스트를 인덱스로 변환
        text = sample['text']
        text_indices = [self.char_to_idx[char] for char in text]
        
        # 길이 제한 및 패딩
        if len(text_indices) > self.max_length:
            text_indices = text_indices[:self.max_length]
        else:
            text_indices = text_indices + [0] * (self.max_length - len(text_indices))
        
        return {
            'image': text_region,
            'text': text,
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'text_length': min(len(text), self.max_length)
        }
    
class InsuranceOCRDataset(Dataset):
    def __init__(self, img_dir, label_dir, char_to_idx=None, transform=None, max_length=25):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.max_length = max_length
        
        # 이미지 파일 목록 가져오기
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 기존 모델의 char_to_idx 사용 또는 새로 생성
        if char_to_idx:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {v: k for k, v in char_to_idx.items()}
        else:
            self.char_list = self._build_char_list()
            self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.char_list)}
            self.char_to_idx['<pad>'] = 0
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def _build_char_list(self):
        # 라벨에서 모든 고유 문자 추출
        chars = set()
        for img_file in self.image_files:
            label_path = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + '.json')
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                for bbox in label_data.get('bbox', []):
                    text = bbox.get('data', '')
                    chars.update(text)
        return sorted(list(chars))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + '.json')
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 라벨 로드
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        # 바운딩 박스의 텍스트 사용
        text = ""
        bbox = None
        if label_data.get('bbox'):
            bbox_item = label_data['bbox'][0]  # 첫 번째 바운딩 박스 사용
            text = bbox_item.get('data', '')
            x_coords = bbox_item.get('x', [0, 0, 0, 0])
            y_coords = bbox_item.get('y', [0, 0, 0, 0])
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        
        # 바운딩 박스가 있으면 이미지 크롭
        if bbox:
            image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # 이미지 변환
        if self.transform:
            image = self.transform(image)
        
        # 텍스트를 인덱스로 변환
        text_indices = [self.char_to_idx.get(c, 0) for c in text]
        text_indices = text_indices[:self.max_length]  # 최대 길이로 자르기
        text_length = len(text_indices)
        
        # 패딩 추가
        text_indices = text_indices + [0] * (self.max_length - len(text_indices))
        
        return {
            'image': image,
            'text': text,
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'text_length': torch.tensor(text_length, dtype=torch.long)
        }
