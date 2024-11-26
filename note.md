#

```powershell
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

cd CosyVoice

# powershell: fsutil.exe file setCaseSensitiveInfo E:\programs\ubuntu\miniconda3\pkgs\ enable 
conda create -n cosyvoice python=3.8 ncurse=6.3
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia  
#或者CPU:  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

## windows有兼容性问题
pip uninstall torchvision
pip install torch==2.4.1 torchaudio==2.4.1

#删掉requirements.txt中torch相关的两个包之后，安装requirements.txt的包
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

#以上环境安装完成之后，启动webui脚本，报错 ImportError: cannot import name 'CommitOperationAdd' from 'huggingface_hub' (/home/hejun/miniconda3/envs/cosyvoice/lib/python3.8/site-packages/huggingface_hub/__init__.py)
conda install -c dglteam dgl

set PYTHONPATH=third_party/Matcha-TTS;%PYTHONPATH%

python webui.py --port 50000 --model_dir /fdata/models/cosyvoice/pretrained_models/CosyVoice-300M-Instruct

```

```python
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
cosyvoice = CosyVoice('/fdata/models/cosyvoice/pretrained_models/CosyVoice-300M-Instruct')
print(cosyvoice.list_avaliable_spks())

for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.', stream=False)):
    print(f"{i} {j}")
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], 22050)
```
