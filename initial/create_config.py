import os

# 문자 집합 읽기
with open('character_set.txt', 'r', encoding='utf-8') as f:
    characters = f.read()

# YAML 설정 파일 생성
yaml_content = f'''
network_params:
  input_channel: 1
  output_channel: 256
  hidden_size: 256

imgH: 64
imgW: 600
batch_size: 32
workers: 4
num_iter: 300000
valInterval: 2000
saved_model: "./pre_trained_model/korean_g2.pth"
FT: True
optimizer: "adam"
lr: 0.0001
freeze: False
character: "{characters} "
'''

with open('user_network_dir/custom.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print("설정 파일이 생성되었습니다.")
