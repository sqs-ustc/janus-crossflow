"""
This file is used to extract feature for visulization during training
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import os
import numpy as np
from tqdm import tqdm
import json
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder

original_sys_path = sys.path.copy()
crossflow_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
janus_dir = os.path.join(crossflow_parent_dir, "Janus")
sys.path.insert(0, os.path.abspath(janus_dir))
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
sys.path = original_sys_path  # 直接恢复

def main():
    prompts = [
        'A road with traffic lights, street lights and cars.',
        'A bus driving in a city area with traffic signs.',
        'A bus pulls over to the curb close to an intersection.',
        'A group of people are walking and one is holding an umbrella.',
        'A baseball player taking a swing at an incoming ball.',
        'A dog next to a white cat with black-tipped ears.',
        'A tiger standing on a rooftop while singing and jamming on an electric guitar under a spotlight. anime illustration.',
        'A bird wearing headphones and speaking into a high-end microphone in a recording studio.',
        'A bus made of cardboard.',
        'A tower in the mountains.',
        'Two cups of coffee, one with latte art of a cat. The other has latter art of a bird.',
        'Oil painting of a robot made of sushi, holding chopsticks.',
        'Portrait of a dog wearing a hat and holding a flag that has a yin-yang symbol on it.',
        'A teddy bear wearing a motorcycle helmet and cape is standing in front of Loch Awe with Kilchurn Castle behind him. dslr photo.',
        'A man standing on the moon',
    ]
    save_dir = f'/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_testset_f10px512/run_vis'
    os.makedirs(save_dir, exist_ok=True)

    render_root_path = '/storage/v-jinpewang/lab_folder/qisheng_data/vis_dataset_f10px512/test_text_render'
    json_path = '/storage/v-jinpewang/lab_folder/qisheng_data/vis_dataset_f10px512/chat_test.json'

    device = 'cuda'
    llm = 't5'  #t5 means janus


    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.half().to(device).eval()
    
    dicts_list = []
    batch_final_tensor = []
    batch_attention_mask = []

    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            dicts_list.append(json.loads(line))

    for i, sample in enumerate(tqdm(dicts_list)):

        question = "Extract the text from the input image with a white background and black text."
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [os.path.join(render_root_path,sample['image'])],
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
        #attention_mask = [1] * len(inputs_embeds[0]) # [576个1]
        attention_mask = [1] * 77
        final_tensor = inputs_embeds.squeeze(dim=0)  # 形状变为 [576, 2048]
        final_tensor = final_tensor[:77,:]
        batch_final_tensor.append(final_tensor.detach().cpu().float().numpy())
        batch_attention_mask.append(attention_mask)

    for i in range(len(prompts)):
        data = {
                'token_embedding': batch_final_tensor[i], 
                'token_mask': batch_attention_mask[i]
                }
        np.save(os.path.join(save_dir, f'{i}.npy'), data)


if __name__ == '__main__':
    main()

