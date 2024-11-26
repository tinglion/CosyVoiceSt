import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

model_path = "/fdata/models/cosyvoice"
# model_path = "/data/models/CosyVoice"
data_path = "./参考音频"

cosyvoice_vc = CosyVoice(f"{model_path}/pretrained_models/CosyVoice-300M-25Hz")
print(cosyvoice_vc.list_avaliable_spks())

# prompt_text = "我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。"
# prompt_speech_16k = load_wav("prompt_st.wav", 16000)
prompt_text = "这天，狐狸挑了一个奇怪的担子，这担子看着像手表，有一个像表针一样的指针，四周有许多小格子。每隔一个空格子，里面就会放上许多好吃好玩儿的东西。狐狸告诉大家，交一块钱就可以拨这个指针，指针指向哪个格子。里面的东西就是他的。"
prompt_speech_16k = load_wav(f"{data_path}/旁白-儿童-示范_30s.mp3", 16000)

tts_text = "这段路程由以色列人的先知摩西带领，通过荒野到西奈山，前往耶和华应许他们的国度 - 迦南地。在西奈山耶和华颁布诫命，典章，律法，及建造会幕的细节，代表着从那时候开始，耶和华从此由天上降临在人间与人同在以及给予他们在进入应许之地的胜利和平安。"
for i, j in enumerate(
    cosyvoice_vc.inference_zero_shot(
        tts_text, prompt_text, prompt_speech_16k, stream=False, speed=0.8
    )
):
    torchaudio.save("zero_{}.wav".format(i), j["tts_speech"], 22050)
