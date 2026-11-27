# -*- coding: utf-8 -*-
"""backend.py --- one array API for both the local CPU smoke test and the
offline GPU run.

    prefer="auto"  -> torch.cuda if available, else torch.cpu if torch exists,
                      else numpy (sandbox / minimal machines)
    prefer="numpy" -> force numpy
    prefer="torch" -> force torch

Every script in this package uses this backend, so the same code can be
validated with `--smoke` before the full run.
"""
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class Backend:
    def __init__(self, prefer="auto", seed=20261120, device=None):
        self.seed = int(seed)
        self.kind = "numpy"
        self.device = "cpu"
        if prefer in ("auto", "torch") and _HAS_TORCH:
            if device is not None:
                self.device = device
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            if prefer == "numpy":
                self.device = "cpu"
            else:
                self.kind = "torch"
                torch.manual_seed(self.seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(self.seed)
                self.gen = torch.Generator(device=self.device)
                self.gen.manual_seed(self.seed)
                self.xp = torch
                return
        # numpy fallback
        self.kind = "numpy"
        self.device = "cpu"
        self.rng = np.random.default_rng(self.seed)
        self.xp = np

    # ---- basic array construction -----------------------------------------
    def asarray(self, x, dtype=None):
        if self.kind == "torch":
            t = torch.as_tensor(x, device=self.device)
            if dtype is not None:
                t = t.to(dtype)
            return t
        return np.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        if self.kind == "torch":
            d = dtype or torch.float32
            return torch.zeros(shape, dtype=d, device=self.device)
        return np.zeros(shape, dtype=dtype or np.float32)

    def ones(self, shape, dtype=None):
        if self.kind == "torch":
            d = dtype or torch.float32
            return torch.ones(shape, dtype=d, device=self.device)
        return np.ones(shape, dtype=dtype or np.float32)

    def arange(self, n):
        if self.kind == "torch":
            return torch.arange(n, device=self.device)
        return np.arange(n)

    def full(self, shape, v, dtype=None):
        if self.kind == "torch":
            d = dtype or torch.float32
            return torch.full(shape, v, dtype=d, device=self.device)
        return np.full(shape, v, dtype=dtype or np.float32)

    # ---- random ------------------------------------------------------------
    def rand(self, shape, dtype=None):
        """U(0,1). `shape` is a tuple."""
        if self.kind == "torch":
            d = dtype or torch.float32
            return torch.rand(shape, generator=self.gen, device=self.device, dtype=d)
        return self.rng.random(shape, dtype=dtype or np.float32)

    def randn(self, shape, dtype=None):
        if self.kind == "torch":
            d = dtype or torch.float32
            return torch.randn(shape, generator=self.gen, device=self.device, dtype=d)
        return self.rng.standard_normal(shape).astype(dtype or np.float32)

    def randint(self, low, high, shape):
        if self.kind == "torch":
            return torch.randint(low, high, shape, generator=self.gen,
                                 device=self.device)
        return self.rng.integers(low, high, size=shape, dtype=np.int64)

    # ---- ops ---------------------------------------------------------------
    def cat(self, xs, axis=1):
        if self.kind == "torch":
            return torch.cat(xs, dim=axis)
        return np.concatenate(xs, axis=axis)

    def log(self, x):
        if self.kind == "torch":
            return torch.log(x)
        return np.log(x)

    def maximum(self, x, y):
        if self.kind == "torch":
            return torch.maximum(x, y)
        return np.maximum(x, y)

    def minimum(self, x, y):
        if self.kind == "torch":
            return torch.minimum(x, y)
        return np.minimum(x, y)

    def quantile(self, x, q):
        if self.kind == "torch":
            return torch.quantile(x, float(q)).item()
        return float(np.quantile(np.asarray(x), q))

    @staticmethod
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        return x.detach().cpu().numpy()

    def info(self):
        if self.kind == "torch":
            name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
            return dict(kind="torch", device=self.device, hardware=name,
                        torch_version=torch.__version__)
        return dict(kind="numpy", device="cpu", hardware="CPU",
                    numpy_version=np.__version__)


def get_backend(prefer="auto", seed=20261120):
    return Backend(prefer=prefer, seed=seed)
