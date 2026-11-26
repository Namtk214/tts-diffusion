"""Executed inside the legacy GenEval environment; stdlib bootstrap only."""
import runpy
import socket
import sys
import os

script, clip_checkpoint, *arguments = sys.argv[1:]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
original_connect = socket.socket.connect
def connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        raise OSError("Offline GenEval: prepare cached checkpoints first")
    return original_connect(sock, address)
socket.socket.connect = connect
import open_clip
from open_clip import openai as openai_loader

def offline_download(url, cache_dir=None):
    if not url.endswith("/ViT-L-14.pt"):
        raise ValueError(f"Unexpected GenEval CLIP checkpoint URL: {url}")
    return clip_checkpoint

openai_loader.download_pretrained_from_url = offline_download
sys.argv = [script, *arguments]
runpy.run_path(script, run_name="__main__")
