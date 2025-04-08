import os
import json
import shutil

# 指定源目录和目标目录
source_root = '/root/autodl-tmp/suqisheng/CrossFlow/text_image_dataset'
target_root = '/root/autodl-tmp/suqisheng/CrossFlow/text_image_dataset_small'

# 指定JSONL文件路径
source_jsonl = os.path.join(source_root, 'img_img_pair.jsonl')
target_jsonl = os.path.join(target_root, 'img_img_pair.jsonl')

# 指定图像文件夹路径
source_imgs = os.path.join(source_root, 'imgs')
source_render_imgs = os.path.join(source_root, 'render_imgs')
target_imgs = os.path.join(target_root, 'imgs')
target_render_imgs = os.path.join(target_root, 'render_imgs')

# 确保目标目录存在
os.makedirs(target_imgs, exist_ok=True)
os.makedirs(target_render_imgs, exist_ok=True)
os.makedirs(os.path.dirname(target_jsonl), exist_ok=True)

# 读取源JSONL文件的前10,000行
with open(source_jsonl, 'r') as f:
    lines = [next(f) for _ in range(10000)]

# 准备目标JSONL文件
with open(target_jsonl, 'w') as f:
    for line in lines:
        f.write(line)

# 复制对应的图像文件
for line in lines:
    data = json.loads(line)
    img_path = data['img_path']
    render_path = data['render_path']

    # 构造源文件路径
    source_img_path = os.path.join(source_imgs, img_path)
    source_render_path = os.path.join(source_render_imgs, render_path)

    # 构造目标文件路径
    target_img_path = os.path.join(target_imgs, img_path)
    target_render_path = os.path.join(target_render_imgs, render_path)

    # 复制图像文件
    shutil.copy2(source_img_path, target_img_path)
    shutil.copy2(source_render_path, target_render_path)

print("任务完成：前10,000行已复制到目标目录，对应的图像文件也已复制。")