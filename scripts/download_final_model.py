"""One-time free open-model download; all training/evaluation then works offline."""
import argparse
from huggingface_hub import snapshot_download
MODEL='HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='work/hf/hub');a=p.parse_args()
 print(snapshot_download(MODEL,revision=REVISION,cache_dir=a.cache,allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE*','README.md']))
if __name__=='__main__':main()
