featureOOM
训练配置config参照了CrossFlow仓库中的follow t2i_256px_clip_dimr.py
在A100上config.train.batch_size可适当调大，其他config路径配置，ckpt下载方式参考CrossFlow原仓库。
janus visencoder输出维度是(576,2048)取77前token，通过(2048->768)的线性层，最终变成(77,768)的feature进行训练。

提取特征pipeline：
extract_test_prompt_feature_nogen.py  提取15条测试prompt，只测前5条(config可配)
输入
render_root_path = '/root/autodl-tmp/suqisheng/CrossFlow/vis_dataset/test_text_render'
json_path = '/root/autodl-tmp/suqisheng/CrossFlow/vis_dataset/chat_test.json'
输出
save_dir = f'/root/autodl-tmp/suqisheng/CrossFlow/vis_dataset/run_vis'

extract_test_feature_1B_nogen.py 通过janus-pro-1B的visencoder提取textimage的feature，janus不进行generate过程（render_root_path装从文字render出来的textimage，json_path装textimager和image的对应关系）
输入
json_path = '/root/autodl-tmp/suqisheng/CrossFlow/text_image_testset/img_img_pair.jsonl'
image_root_path = '/root/autodl-tmp/suqisheng/CrossFlow/text_image_testset/imgs'
render_root_path = '/root/autodl-tmp/suqisheng/CrossFlow/text_image_testset/render_imgs'
输出 
save_dir = f'/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_testset_f10px512/features'

extract_train_feature_1B_nogen.py 通过janus-pro-1B的visencoder提取textimage的feature，janus不进行generate过程（注意提取训练集时间会较长）
输入
json_path = '/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/img_img_pair.jsonl'
image_root_path = '/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/imgs'
render_root_path = '/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/render_imgs'
输出
save_dir = f'/storage/v-jinpewang/lab_folder/qisheng_data/raw_text_image_dataset_f10px512/features'

最终train文件夹包含img_img_pair.jsonl, render_imgs, imgs, features
最终test文件夹包含img_img_pair.jsonl, run_vis（前一个py的这个存过来）, render_imgs, imgs, features
