"""Testbench for Backbone. Run from models/:  python test_backbone.py"""
import time

import torch
import torch.nn as nn

from backbone import Backbone
from layers import DSConv

IN_SHAPE = (1, 1, 160, 160)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def layer_report(model, x):
    """Output shape, params and MACs of every DSConv, via forward hooks."""
    rows, hooks = [], []

    def hook(name):
        def fn(mod, inp, out):
            # MACs: depthwise (9 * C_in * H*W) + pointwise (C_in * C_out * H*W)
            h, w = out.shape[-2:]
            c_in = mod.depth_conv.in_channels
            c_out = mod.point_conv.out_channels
            macs = 9 * c_in * h * w + c_in * c_out * h * w
            rows.append((name, tuple(out.shape), count_params(mod), macs))
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, DSConv):
            hooks.append(mod.register_forward_hook(hook(name)))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()

    print(f"{'layer':<10}{'output shape':<22}{'params':>10}{'MACs':>14}")
    print("-" * 56)
    for name, shape, p, macs in rows:
        print(f"{name:<10}{str(shape):<22}{p:>10,}{macs:>14,}")
    print("-" * 56)
    total_macs = sum(r[3] for r in rows)
    print(f"{'total':<32}{count_params(model):>10,}{total_macs:>14,}")
    print(f"~ {total_macs * 2 / 1e6:.2f} MFLOPs per image\n")
    return rows


def test_output_shape(model):
    y = model(torch.randn(*IN_SHAPE))
    expected = (1, 128, IN_SHAPE[2] // 32, IN_SHAPE[3] // 32)
    assert tuple(y.shape) == expected, f"got {tuple(y.shape)}, expected {expected}"
    print(f"[OK] output shape {tuple(y.shape)}")


def test_batch_sizes(model):
    for b in (1, 4, 16):
        y = model(torch.randn(b, *IN_SHAPE[1:]))
        assert y.shape[0] == b
    print("[OK] batch sizes 1/4/16")


def test_other_resolutions(model):
    for s in (320, 640):
        y = model(torch.randn(1, 1, s, s))
        assert y.shape[-1] == s // 32
    print("[OK] resolutions 320/640")


def test_gradient_flow(model):
    """Every parameter must get a non-zero gradient (catches dead layers)."""
    model.train()
    model.zero_grad()
    model(torch.randn(2, *IN_SHAPE[1:])).mean().backward()
    dead = [n for n, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"no gradient in: {dead}"
    g1 = model.layer1.depth_conv.weight.grad.norm().item()
    g15 = model.layer15.point_conv.weight.grad.norm().item()
    print(f"[OK] gradient flow  |grad| layer1={g1:.2e}  layer15={g15:.2e}")
    model.eval()


def test_eval_deterministic(model):
    model.eval()
    x = torch.randn(*IN_SHAPE)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))
    print("[OK] eval mode deterministic")


def test_activation_stats(model):
    """Checks outputs are not collapsing to zero or exploding."""
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(8, *IN_SHAPE[1:]))
    assert torch.isfinite(y).all(), "NaN/Inf in output"
    print(f"[OK] output stats  mean={y.mean():.3f}  std={y.std():.3f}  "
          f"zeros={(y == 0).float().mean():.1%}")


def benchmark(model, device, runs=100):
    model = model.to(device).eval()
    x = torch.randn(*IN_SHAPE, device=device)
    with torch.no_grad():
        for _ in range(10):  # warm-up
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / runs * 1000
    print(f"[{device}] {ms:.2f} ms/img  ({1000 / ms:.0f} FPS)")


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Backbone().eval()

    layer_report(model, torch.randn(*IN_SHAPE))

    test_output_shape(model)
    test_batch_sizes(model)
    test_other_resolutions(model)
    test_gradient_flow(model)
    test_eval_deterministic(model)
    test_activation_stats(model)

    print()
    benchmark(model, "cpu")
    if torch.cuda.is_available():
        benchmark(model, "cuda")
