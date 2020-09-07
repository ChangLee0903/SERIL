import numpy as np
from pesq import pesq
from pypesq import pesq as pypesq
from pystoi import stoi

def sisdr_eval(src, tar, sr=16000, eps=1e-10):
    alpha = (src * tar).sum() / ((tar * tar).sum() + eps)
    ay = alpha * tar
    norm = ((ay - src) * (ay - src)).sum() + eps
    sisdr = 10 * ((ay * ay).sum() / norm + eps).log10()
    return sisdr.item()

def pesq_nb_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    if np.allclose(src.sum(), 0.0, atol=1e-10) or np.allclose(tar.sum(), 0.0, atol=1e-10):
        print(f'[Evaluation] wav values too small: src {src.sum()}, tar {tar.sum()}')
    mos_lqo = pesq(sr, tar, src/np.abs(src).max(), 'nb')
    return mos_lqo

def pesq_wb_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    if np.allclose(src.sum(), 0.0, atol=1e-10) or np.allclose(tar.sum(), 0.0, atol=1e-10):
        print(f'[Evaluation] wav values too small: src {src.sum()}, tar {tar.sum()}')
    mos_lqo = pesq(sr, tar, src, 'wb')
    return mos_lqo

def pypesq_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    assert not np.allclose(src.sum(), 0.0, atol=1e-6) and not np.allclose(tar.sum(), 0.0, atol=1e-6)
    raw_pesq = pypesq(tar, src, sr)
    return raw_pesq

def stoi_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=False)

def estoi_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=True)