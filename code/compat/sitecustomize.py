"""Compatibility shim (loaded automatically when code/compat is on PYTHONPATH).

torchaudio >= 2.9 reads audio only through the optional `torchcodec` package. If that path fails,
fall back to soundfile so IndicF5's reference-clip loading still works. Audio content is unchanged.
"""
try:
    import torchaudio as _ta

    _orig_load = _ta.load

    def _load(path, *args, **kwargs):
        try:
            return _orig_load(path, *args, **kwargs)
        except Exception:
            import soundfile as _sf
            import torch as _torch
            y, sr = _sf.read(str(path), dtype="float32", always_2d=True)
            return _torch.from_numpy(y.T.copy()), sr

    _ta.load = _load
except Exception:
    pass
