import os
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from models import CRNN
import matplotlib.font_manager as fm
from collections import Counter

# 한글 폰트 설정 개선
font_list = [font.name for font in fm.fontManager.ttflist if 'Gothic' in font.name or '고딕' in font.name]
print("사용 가능한 한글 폰트:", font_list)
plt.rcParams['font.family'] = font_list[0] if font_list else 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 유효한 한글 문자만 포함하도록 필터링하는 함수
def filter_valid_korean(text):
    valid_text = ""
    for c in text:
        # 한글 유니코드 범위 (가-힣) 또는 ASCII 문자
        if (0xAC00 <= ord(c) <= 0xD7A3) or (ord(c) < 128):
            valid_text += c
    return valid_text

# 문자 유형 분류 함수
def classify_char(c):
    code = ord(c)
    if 0xAC00 <= code <= 0xD7A3:  # 한글 (가-힣)
        return 'korean'
    elif 48 <= code <= 57:  # 숫자 (0-9)
        return 'number'
    elif (65 <= code <= 90) or (97 <= code <= 122):  # 영문 (A-Z, a-z)
        return 'english'
    else:
        return 'other'

def calculate_accuracy_by_type(results):
    """
    문자 유형별 정확도를 계산합니다.
    """
    # 유형별 정확도 초기화
    accuracy_by_type = {
        'korean': {'correct': 0, 'total': 0},
        'number': {'correct': 0, 'total': 0},
        'english': {'correct': 0, 'total': 0},
        'other': {'correct': 0, 'total': 0}
    }
    
    # 문자 길이별 정확도 초기화
    accuracy_by_length = {}
    
    for result in results:
        true_text = result['true_text']
        pred_text = result['pred_text']
        
        # 문자 길이별 정확도 계산
        text_len = len(true_text)
        if text_len not in accuracy_by_length:
            accuracy_by_length[text_len] = {'correct': 0, 'total': 0}
        
        accuracy_by_length[text_len]['total'] += 1
        if true_text == pred_text:
            accuracy_by_length[text_len]['correct'] += 1
        
        # 문자 유형별 정확도 계산
        for c in true_text:
            char_type = classify_char(c)
            accuracy_by_type[char_type]['total'] += 1
            
            # 해당 위치의 문자가 예측 텍스트에 있고 동일한지 확인
            idx = true_text.index(c)
            if idx < len(pred_text) and pred_text[idx] == c:
                accuracy_by_type[char_type]['correct'] += 1
    
    # 정확도 계산
    for char_type in accuracy_by_type:
        if accuracy_by_type[char_type]['total'] > 0:
            accuracy_by_type[char_type]['accuracy'] = accuracy_by_type[char_type]['correct'] / accuracy_by_type[char_type]['total']
        else:
            accuracy_by_type[char_type]['accuracy'] = 0
    
    for length in accuracy_by_length:
        if accuracy_by_length[length]['total'] > 0:
            accuracy_by_length[length]['accuracy'] = accuracy_by_length[length]['correct'] / accuracy_by_length[length]['total']
        else:
            accuracy_by_length[length]['accuracy'] = 0
    
    return accuracy_by_type, accuracy_by_length

def visualize_accuracy(accuracy_by_type, accuracy_by_length):
    """
    정확도를 시각화합니다.
    """
    # 문자 유형별 정확도 시각화
    plt.figure(figsize=(12, 5))
    
    # 첫 번째 서브플롯: 문자 유형별 정확도
    plt.subplot(1, 2, 1)
    types = []
    accuracies = []
    counts = []
    
    for char_type in ['korean', 'number', 'english', 'other']:
        if accuracy_by_type[char_type]['total'] > 0:
            types.append(char_type)
            accuracies.append(accuracy_by_type[char_type]['accuracy'] * 100)
            counts.append(accuracy_by_type[char_type]['total'])
    
    bars = plt.bar(types, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # 막대 위에 정확도 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{accuracies[i]:.1f}%\n({counts[i]}개)",
                ha='center', va='bottom')
    
    plt.title('문자 유형별 정확도')
    plt.ylabel('정확도 (%)')
    plt.ylim(0, 110)
    
    # 두 번째 서브플롯: 문자 길이별 정확도
    plt.subplot(1, 2, 2)
    lengths = sorted(accuracy_by_length.keys())
    length_accuracies = [accuracy_by_length[length]['accuracy'] * 100 for length in lengths]
    length_counts = [accuracy_by_length[length]['total'] for length in lengths]
    
    bars = plt.bar(lengths, length_accuracies, color='#9b59b6')
    
    # 막대 위에 정확도와 샘플 수 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{length_accuracies[i]:.1f}%\n({length_counts[i]}개)",
                ha='center', va='bottom', fontsize=8)
    
    plt.title('문자 길이별 정확도')
    plt.xlabel('문자 길이')
    plt.ylabel('정확도 (%)')
    plt.ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('accuracy_analysis.png')
    plt.show()

def visualize_confusion_matrix(results):
    """
    자주 혼동되는 문자 쌍을 시각화합니다.
    """
    confusion_pairs = []
    
    for result in results:
        true_text = result['true_text']
        pred_text = result['pred_text']
        
        # 길이가 다른 경우 짧은 길이까지만 비교
        min_len = min(len(true_text), len(pred_text))
        
        for i in range(min_len):
            if true_text[i] != pred_text[i]:
                confusion_pairs.append((true_text[i], pred_text[i]))
    
    # 가장 자주 혼동되는 쌍 계산
    confusion_counter = Counter(confusion_pairs)
    most_common = confusion_counter.most_common(10)
    
    if not most_common:
        print("혼동되는 문자 쌍이 없습니다.")
        return
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    pairs = [f"'{pair[0][0]}' → '{pair[0][1]}'" for pair in most_common]
    counts = [pair[1] for pair in most_common]
    
    bars = plt.barh(pairs, counts, color='#e74c3c')
    
    # 막대 옆에 빈도 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f"{counts[i]}회",
                ha='left', va='center')
    
    plt.title('가장 자주 혼동되는 문자 쌍')
    plt.xlabel('빈도')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualize_results(results, img_dir, num_samples=5):
    """
    OCR 결과를 시각화합니다.
    """
    # 결과 중 일부만 선택
    samples = results[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        img_path = os.path.join(img_dir, sample['image'])
        img = Image.open(img_path).convert('RGB')
        
        # 바운딩 박스 좌표
        bbox = sample['bbox']
        
        # 텍스트 영역 추출
        text_region = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # 이미지 표시
        axes[i].imshow(text_region)
        
        # 필터링된 텍스트 사용
        filtered_pred = filter_valid_korean(sample['pred_text'])
        
        # 결과 텍스트 표시
        title = f"실제: '{sample['true_text']}' | 예측: '{filtered_pred}'"
        if sample['correct']:
            title += " ✓"
        else:
            title += " ✗"
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('ocr_evaluation_results.png')
    plt.show()

def evaluate_model_direct(model_path, img_dir, label_dir, idx_to_char, sample_ratio=0.1):
    """
    학습된 OCR 모델을 기존 데이터로 직접 평가합니다.
    """
    # 모델 로드
    checkpoint = torch.load(model_path)
    num_chars = len(idx_to_char)
    model = CRNN(num_chars=num_chars)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # GPU 사용 가능 시 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 이미지 전처리 함수
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 평가 지표 초기화
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = 0
    results = []
    
    # 이미지 파일 목록
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 평가에 사용할 샘플 선택
    import random
    sample_size = int(len(image_files) * sample_ratio)
    sampled_files = random.sample(image_files, sample_size)
    
    print(f"총 {len(image_files)} 개의 이미지 중 {sample_size} 개를 평가에 사용합니다.")
    
    for img_file in sampled_files:
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.json')
        
        if not os.path.exists(label_path):
            continue
            
        # 라벨 데이터 로드
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        # 원본 이미지 로드
        full_image = Image.open(img_path).convert('RGB')
        
        # 각 바운딩 박스에 대해 처리
        for bbox_item in label_data.get('bbox', []):
            true_text = bbox_item.get('data', '')
            x_coords = bbox_item.get('x', [0, 0, 0, 0])
            y_coords = bbox_item.get('y', [0, 0, 0, 0])
            
            # 바운딩 박스 좌표
            bbox = [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords)
            ]
            
            # 텍스트 영역 추출
            text_region = full_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            # 전처리 및 모델 입력
            input_tensor = transform(text_region).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                outputs = outputs.permute(1, 0, 2)
                
                # 그리디 디코딩
                _, predicted_indices = torch.max(outputs.permute(1, 0, 2), dim=2)
                
                # CTC 디코딩 (중복 제거 및 빈칸 제거)
                pred_text = ""
                prev_idx = -1
                for idx in predicted_indices[0]:
                    idx = idx.item()
                    if idx != 0 and idx != prev_idx:  # 0은 빈칸(blank)
                        if idx in idx_to_char:
                            pred_text += idx_to_char[idx]
                    prev_idx = idx
                
                # 유효한 한글 문자만 필터링
                pred_text = filter_valid_korean(pred_text)
            
            # 결과 저장
            results.append({
                'image': img_file,
                'bbox': bbox,
                'true_text': true_text,
                'pred_text': pred_text,
                'correct': pred_text == true_text
            })
            
            # 정확도 계산
            if pred_text == true_text:
                correct_words += 1
            total_words += 1
            
            # 문자 단위 정확도
            for a, b in zip(pred_text, true_text):
                if a == b:
                    correct_chars += 1
            total_chars += max(len(pred_text), len(true_text))
    
    # 최종 정확도 계산
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    
    print(f"문자 정확도: {char_accuracy:.4f} ({correct_chars}/{total_chars})")
    print(f"단어 정확도: {word_accuracy:.4f} ({correct_words}/{total_words})")
    
    return char_accuracy, word_accuracy, results

# 사용 예시
if __name__ == "__main__":
    model_path = "models/best_model.pth"
    img_dir = "OCR_sample_data/images"  # 기존 이미지 디렉토리
    label_dir = "OCR_sample_data/labels"  # 기존 라벨 디렉토리
    
    # 모델 로드
    checkpoint = torch.load(model_path)
    idx_to_char = checkpoint['idx_to_char']
    
    # 모델 평가
    char_acc, word_acc, results = evaluate_model_direct(model_path, img_dir, label_dir, idx_to_char, sample_ratio=0.1)
    
    # 결과 시각화 (일부만)
    visualize_results(results[:10], img_dir, num_samples=5)
    
    # 문자 유형별 및 길이별 정확도 분석
    accuracy_by_type, accuracy_by_length = calculate_accuracy_by_type(results)
    visualize_accuracy(accuracy_by_type, accuracy_by_length)
    
    # 혼동 행렬 시각화
    visualize_confusion_matrix(results)
    
    # 오류 분석
    errors = [r for r in results if not r['correct']]
    print(f"\n총 {len(errors)} 개의 오류 발생:")
    
    for i, err in enumerate(errors[:10]):  # 처음 10개 오류만 표시
        filtered_pred = filter_valid_korean(err['pred_text'])
        print(f"{i+1}. 실제: '{err['true_text']}' | 예측: '{filtered_pred}'")
    
    # 정확도 요약 출력
    print("\n=== 정확도 요약 ===")
    print(f"전체 문자 정확도: {char_acc:.4f} ({char_acc*100:.1f}%)")
    print(f"전체 단어 정확도: {word_acc:.4f} ({word_acc*100:.1f}%)")
    
    print("\n문자 유형별 정확도:")
    for char_type in ['korean', 'number', 'english', 'other']:
        if accuracy_by_type[char_type]['total'] > 0:
            acc = accuracy_by_type[char_type]['accuracy']
            total = accuracy_by_type[char_type]['total']
            correct = accuracy_by_type[char_type]['correct']
        print(f" {char_type}: {acc:.4f} ({acc*100:.1f}%) - {correct}/{total}")

print("\n문자 길이별 정확도 (상위 5개):")
sorted_lengths = sorted(accuracy_by_length.items(), key=lambda x: x['total'], reverse=True)[:5]
for length, data in sorted_lengths:
    acc = data['accuracy']
    total = data['total']
    correct = data['correct']
    print(f"  길이 {length}: {acc:.4f} ({acc*100:.1f}%) - {correct}/{total}")