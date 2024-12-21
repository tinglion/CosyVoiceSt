import json
import os
import re
import sys
import test
import time

import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))
# sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))


model_path = "/fdata/models/cosyvoice"
# model_path = "/data/models/CosyVoice"
data_path = "./media"

cosyvoice = CosyVoice(f"{model_path}/pretrained_models/CosyVoice-300M-Instruct")
print(cosyvoice.list_avaliable_spks())

start_time = time.time()
txt = "这段路程由以色列人的先知摩西带领，通过荒野到西奈山，前往耶和华（上帝）应许他们的国度<strong>迦南地</strong>。"
for i, j in enumerate(
    cosyvoice.inference_instruct(
        txt,
        "中文女",
        "A female speaker with normal pitch, slow speaking rate, and happy emotion.",
        stream=False,
        fn_voice="voices/shili.py",
    )
):
    print(f"{i} {j}")
    torchaudio.save("instruct_{}.wav".format(i), j["tts_speech"], 22050)
print(time.time() - start_time)
