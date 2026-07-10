"""
export_aot.py — Export AOT-GAN Places2 generator to ONNX for browser inference.

Usage:
  python export_aot.py --ckpt /path/to/G.pth --repo /path/to/AOT-GAN-for-Inpainting
  python export_aot.py --ckpt G.pth --repo ./AOT-GAN-for-Inpainting --simplify
  python export_aot.py --ckpt G.pth --repo ./AOT-GAN-for-Inpainting --output public/aot_places2.onnx

After export, validate with:
  python inspect_onnx.py public/aot_places2.onnx

Expected output:
  Input:  image_mask  [1, 4, 512, 512]  float32
           Ch 0-2: masked image (RGB normalized to [-1,1], hole pixels = 0)
           Ch 3:   binary mask (1.0 = hole/text, 0.0 = keep)
  Output: output      [1, 3, 512, 512]  float32 in [-1, 1]
  Convert: out_uint8 = clamp((output + 1) / 2 * 255, 0, 255)
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description='Export AOT-GAN Places2 generator to ONNX')
    parser.add_argument('--ckpt',     required=True,  help='Path to Places2 generator checkpoint (G.pth)')
    parser.add_argument('--repo',     required=True,  help='Path to cloned researchmm/AOT-GAN-for-Inpainting')
    parser.add_argument('--output',   default='public/aot_places2.onnx', help='Output ONNX path')
    parser.add_argument('--simplify', action='store_true', help='Run onnx-simplifier after export (pip install onnxsim)')
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f'ERROR: Checkpoint not found: {args.ckpt}')
        sys.exit(1)
    if not os.path.isdir(args.repo):
        print(f'ERROR: Repo directory not found: {args.repo}')
        sys.exit(1)

    # Make both the repo root and its src/ subdirectory importable
    src_path = os.path.join(args.repo, 'src')
    if os.path.isdir(src_path):
        sys.path.insert(0, src_path)
    sys.path.insert(0, args.repo)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print('ERROR: torch not installed — pip install torch torchvision')
        sys.exit(1)

    print(f'PyTorch: {torch.__version__}')

    # Import generator — try both possible import paths from the official repo
    InpaintGenerator = None
    for module_path in ('model.aotgan', 'src.model.aotgan'):
        try:
            import importlib
            mod = importlib.import_module(module_path)
            InpaintGenerator = mod.InpaintGenerator
            print(f'Imported InpaintGenerator from {module_path}')
            break
        except (ImportError, AttributeError):
            continue

    if InpaintGenerator is None:
        print('ERROR: Cannot import InpaintGenerator.')
        print('Make sure --repo points to the root of researchmm/AOT-GAN-for-Inpainting')
        print('and that the repo has src/model/aotgan.py with InpaintGenerator class.')
        sys.exit(1)

    # Build generator with Places2 training defaults
    class DefaultArgs:
        rates = [1, 2, 4, 8]
        block_num = 8

    gen = InpaintGenerator(DefaultArgs())
    gen.eval()

    # Load checkpoint — handles several possible dict formats
    print(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu')

    if isinstance(ckpt, dict):
        for key in ('G', 'generator', 'state_dict', 'model'):
            if key in ckpt:
                state_dict = ckpt[key]
                print(f'Loaded state dict from checkpoint key "{key}"')
                break
        else:
            state_dict = ckpt
            print('Treating entire checkpoint dict as state dict')
    else:
        state_dict = ckpt
        print('Checkpoint is not a dict — using directly as state dict')

    missing, unexpected = gen.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'WARNING: Missing keys ({len(missing)}): {missing[:5]}{"…" if len(missing) > 5 else ""}')
    if unexpected:
        print(f'WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:5]}{"…" if len(unexpected) > 5 else ""}')
    if not missing and not unexpected:
        print('State dict loaded with strict match.')

    # Wrapper: exposes a single [B, 4, H, W] input instead of (image, mask) pair.
    # This matches what the browser worker expects.
    # Ch 0-2: masked RGB (image * (1 - mask)), normalized to [-1, 1]
    # Ch 3:   mask float (1.0 = hole, 0.0 = keep)
    class GeneratorWrapper(nn.Module):
        def __init__(self, g: nn.Module) -> None:
            super().__init__()
            self.g = g

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            return self.g(x[:, :3], x[:, 3:4])

    wrapper = GeneratorWrapper(gen)
    wrapper.eval()

    # Sanity check: forward pass before export
    dummy = torch.zeros(1, 4, 512, 512)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f'Forward test OK: {list(dummy.shape)} → {list(out.shape)}'
          f'  range=[{out.min().item():.3f}, {out.max().item():.3f}]')

    if list(out.shape) != [1, 3, 512, 512]:
        print(f'ERROR: Unexpected output shape {list(out.shape)} — expected [1, 3, 512, 512]')
        sys.exit(1)

    # Create output directory if needed
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f'Exporting to {args.output} (opset 17)…')
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        opset_version=17,
        input_names=['image_mask'],
        output_names=['output'],
        dynamic_axes={
            'image_mask': {0: 'batch'},
            'output':     {0: 'batch'},
        },
    )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'Exported: {args.output} ({size_mb:.1f} MB)')

    # Optional: simplify with onnxsim (reduces op count, can help ORT WASM)
    if args.simplify:
        try:
            import onnx
            import onnxsim
            print('Running onnxsim…')
            model = onnx.load(args.output)
            model_simp, ok = onnxsim.simplify(model)
            if ok:
                onnx.save(model_simp, args.output)
                size_mb2 = os.path.getsize(args.output) / 1024 / 1024
                print(f'Simplified: {size_mb:.1f} MB → {size_mb2:.1f} MB')
            else:
                print('WARNING: onnxsim simplification failed — using unsimplified model')
        except ImportError:
            print('onnxsim not installed — skipping (pip install onnxsim)')

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        print('\nORT verification…')
        sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        inp_info  = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
        out_info  = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
        print(f'  inputs:  {inp_info}')
        print(f'  outputs: {out_info}')

        test_in = np.zeros((1, 4, 512, 512), dtype=np.float32)
        result  = sess.run(None, {sess.get_inputs()[0].name: test_in})
        out_arr = result[0]
        print(f'  inference OK: shape={out_arr.shape}  '
              f'range=[{out_arr.min():.3f}, {out_arr.max():.3f}]')

        if out_arr.shape != (1, 3, 512, 512):
            print(f'  WARNING: Unexpected ORT output shape {out_arr.shape}')
        if out_arr.max() > 1.1 or out_arr.min() < -1.1:
            print(f'  WARNING: Output range outside [-1, 1] — check normalization')
        else:
            print('  Output range is within [-1, 1] ✓')
    except ImportError:
        print('onnxruntime not installed — skipping ORT verification (pip install onnxruntime)')

    print(f'\nDone. Next step:')
    print(f'  python inspect_onnx.py {args.output}')


if __name__ == '__main__':
    main()
