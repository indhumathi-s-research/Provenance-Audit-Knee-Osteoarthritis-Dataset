"""Computational measurement: latency, throughput, FLOPs/MACs, memory, size."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models import build_model, count_parameters
from src.preprocessing import apply_clahe, crop_roi, read_image_rgb, resize_cv


def count_macs(model: nn.Module, input_size) -> int:
    """MACs via forward hooks (conv2d + linear only). Returns total MACs."""
    total = {"macs": 0}

    def hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            n = output.shape[0]
            out_elems = np.prod(output.shape[2:])
            macs = n * out_elems * module.in_channels * np.prod(module.kernel_size) * output.shape[1]
        elif isinstance(module, nn.Linear):
            macs = input[0].shape[0] * module.in_features * module.out_features
        else:
            return
        total["macs"] += int(macs)

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size, device=device))
    for h in hooks:
        h.remove()
    return total["macs"]


def measure_inference_latency(
    model: nn.Module,
    input_size,
    repeats: int = 50,
    warmup: int = 10,
    device: torch.device = None,
) -> dict:
    device = device or next(model.parameters()).device
    model = model.to(device).eval()
    x = torch.randn(input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1000.0  # ms
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "repeats": repeats,
        "warmup": warmup,
        "batch": int(input_size[0]),
    }


def measure_throughput(
    model: nn.Module, input_size, batches: int = 20, device: torch.device = None
) -> dict:
    device = device or next(model.parameters()).device
    model = model.to(device).eval()
    x = torch.randn(input_size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batches):
            with torch.no_grad():
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    n = batches * int(input_size[0])
    return {"images_per_second": float(n / dt), "batch": int(input_size[0]), "batches": batches}


def measure_preprocessing_latency(
    image_path: str,
    size: tuple,
    use_clahe: bool = False,
    use_roi: bool = False,
    repeats: int = 30,
) -> dict:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        img = read_image_rgb(image_path)
        if use_clahe:
            img = apply_clahe(img)
        if use_roi:
            img = crop_roi(img)
        img = resize_cv(img, size)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1000.0
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "repeats": repeats,
        "use_clahe": use_clahe,
        "use_roi": use_roi,
    }


def measure_peak_memory_inference(model: nn.Module, input_size, device: torch.device) -> dict:
    model = model.to(device).eval()
    x = torch.randn(input_size, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)
    if device.type == "cuda":
        return {"peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2))}
    return {"peak_allocated_mb": None, "peak_reserved_mb": None}


def measure_peak_memory_training(model: nn.Module, input_size, device: torch.device, steps: int = 3) -> dict:
    model = model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(input_size, device=device)
    y = torch.randint(0, 5, (input_size[0],), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        return {"peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2))}
    return {"peak_allocated_mb": None, "peak_reserved_mb": None}


def model_size_bytes(model: nn.Module) -> int:
    buf = {}
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    for k, v in state.items():
        buf[k] = v
    n_bytes = sum(v.numel() * v.element_size() for v in buf.values())
    return int(n_bytes)


def measure_roi_latency(image_path: str, repeats: int = 30) -> dict:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        img = read_image_rgb(image_path)
        crop_roi(img)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1000.0
    return {"mean_ms": float(times.mean()), "std_ms": float(times.std()), "repeats": repeats}


def measure_end_to_end_batch1(
    model: nn.Module,
    image_path: str,
    size: tuple,
    device: torch.device,
    use_clahe: bool = False,
    use_roi: bool = False,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    repeats: int = 50,
    warmup: int = 10,
) -> dict:
    """Full pipeline latency: disk read -> preprocess -> normalize -> GPU inference."""
    from src.preprocessing import to_tensor_normalized

    model = model.to(device).eval()

    def once() -> None:
        img = read_image_rgb(image_path)
        if use_clahe:
            img = apply_clahe(img)
        if use_roi:
            img = crop_roi(img)
        img = resize_cv(img, size)
        x = to_tensor_normalized(img, mean, std).unsqueeze(0).to(device)
        with torch.no_grad():
            model(x)

    for _ in range(warmup):
        once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1000.0
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "repeats": repeats,
        "warmup": warmup,
    }


def peak_cpu_memory_bytes() -> int:
    """Peak working set (RAM) of the current process, Windows."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return int(pmc.PeakWorkingSetSize)
    except Exception:
        pass
    return 0


def full_compute_profile(cfg: dict, image_path: str, device: torch.device) -> dict:
    """One-shot compute profile for a configuration (used before training)."""
    model = build_model(cfg)
    input_size = (1, 3) + tuple(cfg["preprocessing"]["image_size"])
    macs = count_macs(model, input_size)
    params = count_parameters(model)
    lat = measure_inference_latency(model, input_size, device=device)
    thpt1 = measure_throughput(model, (1,) + tuple(input_size[1:]), device=device)
    bs = cfg["compute"].get("throughput_batch", 32)
    thpt32 = measure_throughput(model, (bs,) + tuple(input_size[1:]), device=device)
    mem_inf = measure_peak_memory_inference(model, input_size, device)
    mem_tr = measure_peak_memory_training(model, (bs,) + tuple(input_size[1:]), device)
    size_bytes = model_size_bytes(model)
    pre = measure_preprocessing_latency(
        image_path,
        tuple(cfg["preprocessing"]["image_size"]),
        use_clahe=cfg["preprocessing"].get("contrast") == "clahe",
        use_roi=cfg["roi"]["enabled"],
    )
    if cfg["preprocessing"].get("normalize", "raw") == "imagenet":
        e2e_mean, e2e_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        e2e_mean, e2e_std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    e2e = measure_end_to_end_batch1(
        model,
        image_path,
        tuple(cfg["preprocessing"]["image_size"]),
        device,
        use_clahe=cfg["preprocessing"].get("contrast") == "clahe",
        use_roi=cfg["roi"]["enabled"],
        mean=e2e_mean,
        std=e2e_std,
    )
    roi = (
        measure_roi_latency(image_path)
        if cfg["roi"]["enabled"]
        else {"mean_ms": 0.0, "std_ms": 0.0, "repeats": 0}
    )
    return {
        "parameters": params,
        "macs": macs,
        "flops_approx": macs * 2,
        "model_size_bytes": size_bytes,
        "inference_latency_batch1_ms": lat,
        "end_to_end_latency_batch1_ms": e2e,
        "roi_latency_ms": roi,
        "throughput_batch1": thpt1,
        "throughput_batch%d" % bs: thpt32,
        "peak_memory_inference_mb": mem_inf,
        "peak_memory_training_mb": mem_tr,
        "preprocessing_latency_ms": pre,
    }
