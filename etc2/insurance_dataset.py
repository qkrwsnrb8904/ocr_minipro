import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class InsuranceFormDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None, max_len=50):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.max_len = max_len
        
        # 데이터 로드
        self.samples = self._load_dataset()
        
        # 문자 집합 생성
        self.char_set = set()
        for sample in self.samples:
            for region in sample['text_regions']:
                for char in region['text']:
                    self.char_set.add(char)
        
        # 문자를 인덱스로 매핑
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(list(self.char_set)))}
        self.char_to_idx['<blank>'] = 0  # CTC loss를 위한 blank 토큰
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # 모든 텍스트 영역을 평탄화
        self.flattened_samples = []
        for sample in self.samples:
            for region in sample['text_regions']:
                self.flattened_samples.append({
                    'image_path': sample['image_path'],
                    'text': region['text'],
                    'points': region['points']
                })
        
        print(f"총 샘플 수: {len(self.flattened_samples)}")
        print(f"문자 집합 크기: {len(self.char_set)}")
    
    def _load_dataset(self):
        data = []
        for json_file in os.listdir(self.annotation_folder):
            if json_file.endswith('.json'):
                json_path = os.path.join(self.annotation_folder, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    
                    # 이미지 파일명 추출 (확장자를 .png로 설정)
                    if 'Images' in annotation and 'identifier' in annotation['Images']:
                        image_filename = annotation['Images']['identifier'] + '.png'
                    else:
                        image_filename = json_file.replace('.json', '.png')
                    
                    image_path = os.path.join(self.image_folder, image_filename)
                    
                    if not os.path.exists(image_path):
                        # 다른 확장자 시도
                        for ext in ['.jpg', '.jpeg']:
                            alt_image_path = os.path.join(self.image_folder, image_filename.replace('.png', ext))
                            if os.path.exists(alt_image_path):
                                image_path = alt_image_path
                                break
                        else:
                            print(f"이미지 파일을 찾을 수 없습니다: {image_filename}")
                            continue
                    
                    # 텍스트 영역 정보 추출
                    text_regions = []
                    
                    # AI Hub 보험 신청서 데이터셋 구조에 맞게 수정
                    if 'bbox' in annotation:
                        for bbox_item in annotation['bbox']:
                            if 'data' in bbox_item and 'x' in bbox_item and 'y' in bbox_item:
                                text = bbox_item['data']
                                x_coords = bbox_item['x']
                                y_coords = bbox_item['y']
                                
                                # x, y 좌표를 points 형식으로 변환
                                points = []
                                for i in range(len(x_coords)):
                                    points.append([x_coords[i], y_coords[i]])
                                
                                # 빈 텍스트 건너뛰기
                                if not text or text.isspace():
                                    continue
                                
                                text_regions.append({
                                    'text': text,
                                    'points': points
                                })
                    
                    if text_regions:
                        data.append({
                            'image_path': image_path,
                            'text_regions': text_regions
                        })
                except Exception as e:
                    print(f"파일 처리 중 오류 발생: {json_file}, 오류: {str(e)}")
        
        return data
    
    def __len__(self):
        return len(self.flattened_samples)
    
    def __getitem__(self, idx):
        sample = self.flattened_samples[idx]
        image_path = sample['image_path']
        text = sample['text']
        points = sample['points']
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            image = np.zeros((32, 128, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 텍스트 영역 추출
        points_array = np.array(points, dtype=np.int32)
        x_min, y_min = points_array.min(axis=0)
        x_max, y_max = points_array.max(axis=0)
        
        # 여백 추가
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # 텍스트 영역 크롭
        cropped = image[y_min:y_max, x_min:x_max]
        
        # 이미지 전처리 (크기 조정)
        try:
            cropped = cv2.resize(cropped, (128, 32))  # CRNN 모델 입력 크기에 맞게 조정
        except Exception as e:
            print(f"이미지 리사이즈 오류: {e}, 이미지 크기: {cropped.shape if cropped is not None else 'None'}")
            # 오류 발생 시 빈 이미지 생성
            cropped = np.zeros((32, 128, 3), dtype=np.uint8)
        
        # 변환 적용
        if self.transform:
            cropped = self.transform(cropped)
        
        # 텍스트를 인덱스 시퀀스로 변환
        target = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        target_length = len(target)
        
        # 최대 길이 제한
        if len(target) > self.max_len:
            target = target[:self.max_len]
            target_length = self.max_len
        
        return cropped, torch.tensor(target), target_length

# 데이터 로더를 위한 collate 함수
def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths)
    return images, targets, target_lengths
