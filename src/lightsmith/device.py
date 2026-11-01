"""CPU/GPU device selection via the Python array API standard.

Native solvers are written once against the array API (through
``array_api_compat``) and run on NumPy (CPU), PyTorch (CPU/CUDA/ROCm) or CuPy
(CUDA). Install a GPU framework with ``uv sync --extra gpu``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _torch_cuda() -> bool:
    if not _has("torch"):
        return False
    import torch

    return torch.cuda.is_available()


def _cupy_cuda() -> bool:
    if not _has("cupy"):
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@dataclass(frozen=True)
class Device:
    framework: str  # "numpy" | "torch" | "cupy"
    name: str  # "cpu" | "cuda:0" | ...

    @property
    def is_gpu(self) -> bool:
        return self.name != "cpu"

    def namespace(self) -> Any:
        if self.framework == "numpy":
            import array_api_compat.numpy as xp
        elif self.framework == "torch":
            import array_api_compat.torch as xp
        else:
            import array_api_compat.cupy as xp
        return xp

    def native_device(self) -> Any:
        if self.framework == "torch":
            import torch

            return torch.device(self.name)
        if self.framework == "cupy":
            import cupy

            return cupy.cuda.Device(int(self.name.split(":")[1]) if ":" in self.name else 0)
        return "cpu"

    def asarray(self, a, dtype=None):
        xp = self.namespace()
        dt = getattr(xp, dtype) if isinstance(dtype, str) else dtype
        if self.framework == "cupy":
            with self.native_device():
                return xp.asarray(a, dtype=dt)
        return xp.asarray(a, dtype=dt, device=self.native_device())

    @staticmethod
    def to_numpy(a) -> np.ndarray:
        if hasattr(a, "detach"):
            return a.detach().cpu().numpy()
        if hasattr(a, "get"):
            return a.get()
        return np.asarray(a)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"framework": self.framework, "device": self.name}
        if self.framework == "torch" and self.is_gpu:
            import torch

            d["gpu_name"] = torch.cuda.get_device_name(torch.device(self.name))
        return d


def resolve_device(device: str = "auto") -> Device:
    """Map a user-facing device string to a concrete :class:`Device`.

    ``"auto"`` picks a CUDA GPU when one is usable, else CPU.
    ``"gpu"``/``"cuda"``/``"cuda:N"`` require a GPU and raise with install hints otherwise.
    ``"torch-cpu"`` runs the native solvers through PyTorch on the CPU.
    """
    device = device.lower()
    if device == "cpu":
        return Device("numpy", "cpu")
    if device == "torch-cpu":
        if not _has("torch"):
            raise RuntimeError("PyTorch is not installed; run `uv sync --extra gpu`.")
        return Device("torch", "cpu")
    if device in ("auto", "gpu") or device.startswith("cuda"):
        name = "cuda:0" if device in ("auto", "gpu", "cuda") else device
        if _torch_cuda():
            return Device("torch", name)
        if _cupy_cuda():
            return Device("cupy", name)
        if device == "auto":
            return Device("numpy", "cpu")
        raise RuntimeError(
            "No usable CUDA GPU found. Install a GPU array framework with `uv sync --extra gpu` "
            "(PyTorch CUDA build) and check `lightsmith doctor`."
        )
    raise ValueError(f"unknown device {device!r}; use 'auto', 'cpu', 'gpu', 'cuda:N' or 'torch-cpu'")


def doctor() -> dict[str, Any]:
    """Report which array frameworks and GPUs are usable. Safe to call anywhere."""
    info: dict[str, Any] = {"numpy": np.__version__, "torch": None, "cupy": None, "cuda_gpus": []}
    if _has("torch"):
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if _has("cupy"):
        import cupy

        info["cupy"] = cupy.__version__
    info["auto_device"] = resolve_device("auto").describe()
    return info
