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
original_create = open_clip.create_model_and_transforms
def create(*args, **kwargs):
    if kwargs.get("pretrained") == "openai":
        kwargs["pretrained"] = clip_checkpoint
    return original_create(*args, **kwargs)
open_clip.create_model_and_transforms = create
sys.argv = [script, *arguments]
runpy.run_path(script, run_name="__main__")
