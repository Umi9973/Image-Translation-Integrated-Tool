"""
Phase 0 — ONNX model inspection script.
Usage: python inspect_onnx.py <path-to-model.onnx>
Run from the project root with the .venv active, or call directly:
  .venv/Scripts/python.exe inspect_onnx.py public/aot_inpaint.onnx
"""

import sys
import numpy as np
import onnxruntime as ort

def inspect(path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Model: {path}")
    print(f"ORT version: {ort.__version__}")
    print(f"{'='*60}\n")

    # ── Load session ──────────────────────────────────────────────
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # ── Input metadata ────────────────────────────────────────────
    print("INPUTS")
    print("-" * 40)
    inputs = sess.get_inputs()
    for inp in inputs:
        print(f"  name  : {inp.name}")
        print(f"  shape : {inp.shape}")
        print(f"  dtype : {inp.type}")
        print()

    # ── Output metadata ───────────────────────────────────────────
    print("OUTPUTS")
    print("-" * 40)
    outputs = sess.get_outputs()
    for out in outputs:
        print(f"  name  : {out.name}")
        print(f"  shape : {out.shape}")
        print(f"  dtype : {out.type}")
        print()

    # ── Determine likely input size ───────────────────────────────
    # Shape entries may be ints or symbolic strings (dynamic dims).
    def resolve_dim(d, fallback: int) -> int:
        return d if isinstance(d, int) and d > 0 else fallback

    first = inputs[0]
    shape = first.shape          # e.g. [1, 3, 512, 512] or [1, 3, 'h', 'w']
    in_channels = resolve_dim(shape[1] if len(shape) >= 2 else 0, 3)
    in_H        = resolve_dim(shape[2] if len(shape) >= 3 else 0, 512)
    in_W        = resolve_dim(shape[3] if len(shape) >= 4 else 0, 512)

    print(f"Resolved test size: {in_channels}×{in_H}×{in_W}")
    print(f"Input count: {len(inputs)}")

    if len(inputs) == 1 and in_channels == 4:
        print("→ Likely 4-channel RGBA or image+mask concatenated (single tensor)")
    elif len(inputs) == 2:
        print("→ Likely separate image + mask tensors")
    elif len(inputs) == 1 and in_channels == 3:
        print("→ Single RGB image tensor (mask may be implicit or not used)")
    else:
        print("→ Unknown layout — inspect shape manually")

    # ── Dummy inference ───────────────────────────────────────────
    print("\nDUMMY INFERENCE")
    print("-" * 40)

    feeds = {}
    for inp in inputs:
        shape = inp.shape
        # Replace dynamic dims with the resolved values
        concrete = []
        for i, d in enumerate(shape):
            if i == 0:
                concrete.append(1)          # batch = 1
            elif i == 1:
                concrete.append(resolve_dim(d, in_channels))
            elif i == 2:
                concrete.append(resolve_dim(d, in_H))
            elif i == 3:
                concrete.append(resolve_dim(d, in_W))
            else:
                concrete.append(resolve_dim(d, 1))

        dtype_map = {
            "tensor(float)":  np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)":  np.int64,
            "tensor(uint8)":  np.uint8,
        }
        np_dtype = dtype_map.get(inp.type, np.float32)

        # For image input fill with mid-grey (0.5 for 0-1 models, 128 for 0-255 models)
        # For mask input fill with 1s (regenerate everything — stress test)
        is_mask = "mask" in inp.name.lower()
        fill    = 1.0 if is_mask else 0.5
        dummy   = np.full(concrete, fill, dtype=np_dtype)
        feeds[inp.name] = dummy
        print(f"  feed '{inp.name}': shape={concrete} dtype={np_dtype.__name__} fill={fill}")

    results = sess.run(None, feeds)
    print()

    for out, arr in zip(outputs, results):
        arr = np.array(arr)
        print(f"  output '{out.name}':")
        print(f"    shape : {arr.shape}")
        print(f"    dtype : {arr.dtype}")
        print(f"    min   : {arr.min():.4f}")
        print(f"    max   : {arr.max():.4f}")
        print(f"    mean  : {arr.mean():.4f}")
        print()

    # ── Normalization guess ───────────────────────────────────────
    print("NORMALIZATION GUESS")
    print("-" * 40)
    out_arr = np.array(results[0])
    out_max = out_arr.max()
    if out_max <= 1.05:
        print("  Output range ≈ 0–1  →  model likely expects 0–1 normalized input")
        print("  Preprocessing: pixel / 255.0")
        print("  Postprocessing: output × 255, clip to [0, 255]")
    elif out_max <= 255.5:
        print("  Output range ≈ 0–255  →  model likely expects 0–255 raw input (like LaMa)")
        print("  Preprocessing: raw uint8 cast to float32")
        print("  Postprocessing: clip to [0, 255], cast to uint8")
    else:
        print(f"  Output max = {out_max:.1f}  →  unusual range, check model docs")

    # ── Mask convention test hint ─────────────────────────────────
    print()
    print("MASK CONVENTION")
    print("-" * 40)
    print("  Dummy inference used mask fill=1.0 (all pixels marked as regenerate).")
    print("  If output looks like a plausible reconstruction → 1=regenerate (same as LaMa).")
    print("  If output looks unchanged or black → mask convention likely inverted (1=keep).")
    print("  → To verify: run again with mask fill=0.0 and compare outputs.")

    print(f"\n{'='*60}")
    print("Inspection complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_onnx.py <path-to-model.onnx>")
        sys.exit(1)
    inspect(sys.argv[1])
