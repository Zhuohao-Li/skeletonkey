from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "garrick0508/vision_mamba"

model_files = [
    "output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/checkpoint.pth",
    "output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/checkpoint.pth",
    # 添加其他文件...
]

for file in model_files:
    print(f"Uploading {file}...")
    api.upload_file(
        path_or_fileobj=f"./{file}",
        path_in_repo=file,
        repo_id=repo_id
    )