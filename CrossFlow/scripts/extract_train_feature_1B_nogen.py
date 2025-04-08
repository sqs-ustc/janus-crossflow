"""
This file is used to extract feature of the demo training data.
"""

import os
import shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import io
import einops
import random
import json
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder
from transformers import AutoModelForCausalLM

print("sys.path:",sys.path)
original_sys_path = sys.path.copy()
crossflow_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
janus_dir = os.path.join(crossflow_parent_dir, "Janus")
# sys.path.insert(0, os.path.abspath(janus_dir))
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
sys.path = original_sys_path  # 直接恢复

def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main(bz = 4):
    device = "cuda:7"

    json_path = '/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/img_img_pair.json'
    image_root_path = '/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/imgs'
    render_root_path = '/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/render_imgs'
    # specify the path to the model
    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.half().to(device).eval()


    dicts_list = []
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            dicts_list.append(json.loads(line))

    save_dir = f'/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/features'
    
    recreate_folder(save_dir)

    autoencoder = libs.autoencoder.get_model('/storage/v-jinpewang/lab_folder/qisheng_data/assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)


    idx = 0
    batch_img_256 = []
    batch_img_512 = []
    batch_caption = []
    batch_render_img_256 = []
    batch_render_img_512 = []
    batch_final_tensor = []
    batch_attention_mask = []
    batch_name = []
    print("len(dicts_list):", len(dicts_list))
    for i, sample in enumerate(tqdm(dicts_list)):
        try:
            pil_image = Image.open(os.path.join(image_root_path,sample['img_path']))
            # caption = sample['prompt']
            render_image = Image.open(os.path.join(render_root_path,sample['render_path']))
            img_name = sample['img_path'].replace('.jpg','')
            
            pil_image.load()
            pil_image = pil_image.convert("RGB")
        except:
            with open("failed_file.txt", 'a+') as file: 
                file.write(sample['img_path'] + "\n")
            continue

        image_256 = center_crop_arr(pil_image, image_size=256)
        image_512 = center_crop_arr(pil_image, image_size=512)


        image_256 = (image_256 / 127.5 - 1.0).astype(np.float32)
        image_256 = einops.rearrange(image_256, 'h w c -> c h w')
        batch_img_256.append(image_256)

        image_512 = (image_512 / 127.5 - 1.0).astype(np.float32)
        image_512 = einops.rearrange(image_512, 'h w c -> c h w')
        batch_img_512.append(image_512)

        batch_name.append(img_name)
    
        # question = "Extract the text from the input image with a white background and black text."
        question = ""
        conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [os.path.join(render_root_path,sample['render_path'])],
        },
        {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(device)

        # # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)  #modeling_vlm.py changed! only return image_embeds

        #inputs_embeds.shape:torch.Size([1, 576, 2048])
        attention_mask = [1] * len(inputs_embeds[0]) # [576个1]
        final_tensor = inputs_embeds.squeeze(dim=0)  # 形状变为 [576, 2048]
        batch_final_tensor.append(final_tensor.detach().cpu().float().numpy())
        batch_attention_mask.append(attention_mask)

        if len(batch_name) == bz or i == len(dicts_list) - 1:
            batch_img_256 = torch.tensor(np.stack(batch_img_256)).to(device)
            moments_256 = autoencoder(batch_img_256, fn='encode_moments').squeeze(0)
            moments_256 = moments_256.detach().cpu().numpy()

            batch_img_512 = torch.tensor(np.stack(batch_img_512)).to(device)
            moments_512 = autoencoder(batch_img_512, fn='encode_moments').squeeze(0)
            moments_512 = moments_512.detach().cpu().numpy()

            # _latent_clip, latent_and_others_clip = clip.encode(batch_caption)
            # token_embedding_clip = latent_and_others_clip['token_embedding'].detach().cpu().numpy()
            # token_mask_clip = latent_and_others_clip['token_mask'].detach().cpu().numpy()
            # token_clip = latent_and_others_clip['tokens'].detach().cpu().numpy()

            token_embedding_janus = batch_final_tensor
            # mask: 1 for tokens that are not masked, 0 for tokens that are masked.
            mask_janus = batch_attention_mask 
            token_janus = None

            # _latent_t5, latent_and_others_t5 = t5.get_text_embeddings(batch_caption)
            # token_embedding_t5 = (latent_and_others_t5['token_embedding'].to(torch.float32) * 10.0).detach().cpu().numpy()
            # token_mask_t5 = latent_and_others_t5['token_mask'].detach().cpu().numpy()
            # token_t5 = latent_and_others_t5['tokens'].detach().cpu().numpy()
            token_embedding_clip = []
            token_mask_clip = []
            token_clip = []
            batch_final_tensor = []
            batch_attention_mask = []
            
            for mt_256, mt_512, te_t, tm_t, bn in zip(moments_256, moments_512, token_embedding_janus, mask_janus, batch_name):
                assert mt_256.shape == (8,32,32)
                assert mt_512.shape == (8,64,64)
                assert te_t.shape == (576, 2048)  #modified
                tar_path_name = os.path.join(save_dir, f'{bn}.npy')
                if os.path.exists(tar_path_name):
                    os.remove(tar_path_name)
                data = {'image_latent_256': mt_256,
                        'image_latent_512': mt_512,
                        'token_embedding_clip': None, 
                        'token_embedding_t5': te_t,  #janus提取的text image embedding
                        'token_mask_clip': None,
                        'token_mask_t5': tm_t,
                        'token_clip': None,
                        'token_t5': None,
                        'batch_caption': None}
                try:
                    np.save(tar_path_name, data)
                    idx += 1
                except:
                    pass
                    print(f'{idx} error')
            
            batch_img_256 = []
            batch_img_512 = []
            batch_caption = []
            batch_name = []

    print(f'save {idx} files')

if __name__ == '__main__':
    main()

