import json

# JSON 파일 경로 설정
json_file_path = "보험_확인서\라벨\IMG_OCR_6_F_0174217.json"

# JSON 파일 읽기
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# bbox 섹션에서 ID와 텍스트 추출
for bbox in data.get("bbox", []):
    text = bbox.get("data")
    bbox_id = bbox.get("id")
    print(f"ID: {bbox_id}, Text: {text}")
