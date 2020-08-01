import torch
from pypesq import pesq
from pystoi import stoi

def pesq_eval(src, tar, sr=16000):
    src = src.squeeze().cpu().detach().numpy()
    tar = tar.squeeze().cpu().detach().numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return pesq(tar, src, sr)

def stoi_eval(src, tar, sr=16000):
    src = src.squeeze().cpu().detach().numpy()
    tar = tar.squeeze().cpu().detach().numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=False)

def estoi_eval(src, tar, sr=16000):
    src = src.squeeze().cpu().detach().numpy()
    tar = tar.squeeze().cpu().detach().numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=True)