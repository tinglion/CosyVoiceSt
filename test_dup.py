import time

import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

model_path = "/fdata/models/cosyvoice"
# model_path = "/data/models/CosyVoice"
data_path = "./参考音频"

cosyvoice_vc = CosyVoice(f"{model_path}/pretrained_models/CosyVoice-300M")
print(cosyvoice_vc.list_avaliable_spks())

prompt_speech_16k = load_wav(f"{data_path}/旁白-儿童-示范_30s.wav", 16000)
source_speech_16k = load_wav(f"{data_path}/paopaofeichuan_30s.wav", 16000)

start_time = time.time()
for i, j in enumerate(
    cosyvoice_vc.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)
):
    torchaudio.save("vc_{}.wav".format(i), j["tts_speech"], 22050)
print(time.time() - start_time)
