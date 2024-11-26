# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['MODELSCOPE_CACHE'] ='./.cache/modelscope'
# os.environ['TORCH_HOME'] = './.cache/torch'  #设置torch的缓存目录
# os.environ["HF_HOME"] = "./.cache/huggingface" #设置transformer的缓存目录
# os.environ['XDG_CACHE_HOME']="./.cache"
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import shutil

import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa


import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


reference_wavs = ["请选择参考音频或者自己上传"]
for name in os.listdir("./参考音频/"):
    reference_wavs.append(name)

spk_new = ["无"]

for name in os.listdir("./voices/"):
    print(name.replace(".py",""))
    spk_new.append(name.replace(".py",""))


def refresh_choices():

    spk_new = ["无"]

    for name in os.listdir("./voices/"):
        print(name.replace(".py",""))
        spk_new.append(name.replace(".py",""))
    
    return {"choices":spk_new, "__type__": "update"}


def change_choices():

    reference_wavs = ["选择参考音频,或者自己上传"]

    for name in os.listdir("./参考音频/"):
        reference_wavs.append(name)
    
    return {"choices":reference_wavs, "__type__": "update"}


def change_wav(audio_path):

    text = audio_path.replace(".wav","").replace(".mp3","")

    return f"./参考音频/{audio_path}",text


def save_name(name):

    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False

    shutil.copyfile("./output.py",f"./voices/{name}.py")
    gr.Info("音色保存成功,存放位置为voices目录")

    

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2.点击生成音频按钮',
                 '3s极速复刻': '1. 选择参考音频文件，或录入参考音频，若同时提供，优先选择参考音频文件\n2. 输入参考文本\n3.点击生成音频按钮',
                 '跨语种复刻': '1. 选择参考音频文件，或录入参考音频，若同时提供，优先选择参考音频文件\n2.点击生成音频按钮',
                 '自然语言控制': '1. 输入instruct文本\n2.点击生成音频按钮'}
def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed,new_dropdown):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is speech_tts/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用speech_tts/CosyVoice-300M-Instruct模型'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, 参考音频/参考文本会被忽略')
    # if cross_lingual mode, please make sure that model is speech_tts/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用speech_tts/CosyVoice-300M模型'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供参考音频')
            return (target_sr, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和参考文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入参考音频？')
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('参考音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，参考文本/参考音频/instruct文本会被忽略！')
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入参考文本？')
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text,sft_dropdown,new_dropdown)
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text)
    audio_data = output['tts_speech'].numpy().flatten()
    return (target_sr, audio_data)

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### CosyVoice - 阿里开源情感语音克隆、文本生成语音项目，支持粤语")
        gr.Markdown("#### 更多好玩的AI应用，访问 https://deepface.cc")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我轻轻的我走了，正如我轻轻地来。我挥一挥衣袖，不带走一片云彩。")

        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            new_dropdown = gr.Dropdown(choices=spk_new, label='选择新增音色', value=spk_new[0],interactive=True)
            refresh_new_button = gr.Button("刷新新增音色")
            refresh_new_button.click(fn=refresh_choices, inputs=[], outputs=[new_dropdown])
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            wavs_dropdown = gr.Dropdown(label="参考音频列表",choices=reference_wavs,value="请选择参考音频或者自己上传",interactive=True)
            refresh_button = gr.Button("刷新参考音频")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[wavs_dropdown])
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择参考音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制参考音频文件')
        prompt_text = gr.Textbox(label="输入参考文本", lines=1, placeholder="请输入参考文本，需与参考音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        new_name = gr.Textbox(label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value='')

        save_button = gr.Button("保存刚刚推理的zero-shot音色")

        save_button.click(save_name, inputs=[new_name])

        wavs_dropdown.change(change_wav,[wavs_dropdown],[prompt_wav_upload,prompt_text])

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频")

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed,new_dropdown],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_port=args.port,inbrowser=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='speech_tts/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
