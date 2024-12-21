import time
from typing import Optional

import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

model_path = "/fdata/models/cosyvoice"
# model_path = "/data/models/CosyVoice"
data_path = "./media"

cosyvoice_vc = CosyVoice(f"{model_path}/pretrained_models/CosyVoice-300M")
print(cosyvoice_vc.list_avaliable_spks())


app = FastAPI()


class VcReq(BaseModel):
    src: str = "/projects/ady/tts/CosyVoice/media/e.wav"
    voice_color_file: str = "/projects/ady/tts/CosyVoice/media/旁白-儿童-示范_30s.wav"


@app.get("/")
async def hi():
    return {"Hi": "Yuki"}


# 获取所有项目的端点
@app.post("/vc")
async def vc(req: VcReq):
    data = []
    print(f"req={req}")

    start_time = time.time()
    src_speech_16k = load_wav(req.src, 16000)
    # 音色
    prompt_speech_16k = load_wav(req.voice_color_file, 16000)
    for i, j in enumerate(
        cosyvoice_vc.inference_vc(
            src_speech_16k,
            prompt_speech_16k,
            stream=False,
        )
    ):
        torchaudio.save(f"{req.src}_{i}.wav", j["tts_speech"], 22050)
    print(time.time() - start_time)
    return dict(code=200, data=data)


# 运行应用
# uvicorn server_cosyvoice:app --reload --host 0.0.0.0 --port 16000
