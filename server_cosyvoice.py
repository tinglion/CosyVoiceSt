import time
from typing import Optional

import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from conf import model_path,data_path

cosyvoice_vc = CosyVoice(f"{model_path}/pretrained_models/CosyVoice-300M")
print(cosyvoice_vc.list_avaliable_spks())


app = FastAPI()


class VcReq(BaseModel):
    src: str = f"{data_path}/旁白-儿童-示范_30s.wav"
    dst: str = f"{data_path}/旁白-儿童-示范_30s.brian.wav"
    voice_color_file: str = f"{data_path}/11labs/brian.wav"


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
        torchaudio.save(f"{req.dst}", j["tts_speech"], 22050)
    print(f"@sting {time.time() - start_time}")
    return dict(code=200, data=data)


# 运行应用 --reload 
# nohup uvicorn server_cosyvoice:app --host 0.0.0.0 --port 16000 >>logs/info.log 2>&1 &
