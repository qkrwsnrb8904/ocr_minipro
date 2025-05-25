import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class InsuranceFormDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, char_to_idx, transform=None, max_len=50):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.max_len = max_len
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # 데이터 로드
        self.samples = self._load_dataset()
        print(f"로드된 샘플 수: {len(self.samples)}")
    
    def _load_dataset(self):
        samples = []
        for json_file in os.listdir(self.annotation_folder):
            if json_file.endswith('.json'):
                json_path = os.path.join(self.annotation_folder, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    
                    # 이미지 파일명 (확장자는 .png로 가정)
                    image_filename = annotation.get("Images", {}).get("identifier", "")
                    if not image_filename:
                        image_filename = os.path.splitext(json_file)[0]
                    
                    image_path = os.path.join(self.image_folder, f"{image_filename}.png")
                    if not os.path.exists(image_path):
                        # 다른 확장자 시도
                        for ext in ['.jpg', '.jpeg']:
                            alt_path = os.path.join(self.image_folder, f"{image_filename}{ext}")
                            if os.path.exists(alt_path):
                                image_path = alt_path
                                break
                        else:
                            continue
                    
                    # 바운딩 박스 정보 추출
                    for bbox in annotation.get("bbox", []):
                        text = bbox.get("data", "")
                        if not text:
                            continue
                        
                        x_coords = bbox.get("x", [])
                        y_coords = bbox.get("y", [])
                        if not x_coords or not y_coords:
                            continue
                        
                        samples.append({
                            "image_path": image_path,
                            "text": text,
                            "x_coords": x_coords,
                            "y_coords": y_coords
                        })
                except Exception as e:
                    print(f"파일 처리 중 오류: {json_file}, {str(e)}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        text = sample["text"]
        x_coords = sample["x_coords"]
        y_coords = sample["y_coords"]
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            # 이미지 로드 실패 시 빈 이미지 생성
            image = np.zeros((32, 128, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 바운딩 박스 영역 추출
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 여백 추가
            padding = 5
            height, width = image.shape[:2]
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            # 영역 크롭
            try:
                image = image[y_min:y_max, x_min:x_max]
                # 크기 조정
                image = cv2.resize(image, (128, 32))
            except Exception as e:
                print(f"이미지 처리 오류: {str(e)}")
                image = np.zeros((32, 128, 3), dtype=np.uint8)
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # 텍스트를 인덱스로 변환
        target = []
        for char in text:
            if char in self.char_to_idx:
                target.append(self.char_to_idx[char])
            else:
                # 알 수 없는 문자는 0(blank)으로 처리
                target.append(0)
        
        # 최대 길이 제한
        if len(target) > self.max_len:
            target = target[:self.max_len]
        
        return image, torch.tensor(target), len(target)

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths)
    return images, targets, target_lengths
