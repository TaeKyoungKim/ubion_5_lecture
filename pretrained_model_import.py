from huggingface_hub import hf_hub_download

# FFHQ 얼굴 모델 (1024x1024)
hf_hub_download(
    repo_id='nvidia/stylegan2-ada-pytorch',
    filename='ffhq.pkl',
    local_dir='./pretrained'
)
print('✓ 다운로드 완료: ./pretrained/ffhq.pkl')