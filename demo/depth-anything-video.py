#!/usr/bin/env python3
"""
demo/depth-anything-video.py

Usage:
  python demo/depth-anything-video.py --input test1.MP4 --output depth_out.mp4 --colormap plasma

Or for webcam:
  python demo/depth-anything-video.py --camera 0 --output depth_cam.mp4 --colormap plasma
"""
import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

def normalize_to_uint8(arr):
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return (np.clip(arr, 0, 1) * 255).astype('uint8')
    norm = (arr - mn) / (mx - mn)
    return (norm * 255).astype('uint8')

def apply_colormap(norm_arr, cmap_name='plasma'):
    # norm_arr: 2D numpy array in 0..1 (float) or 0..255 (uint8)
    import matplotlib.cm as cm
    if norm_arr.dtype == np.uint8:
        norm_f = norm_arr.astype('float32') / 255.0
    else:
        norm_f = norm_arr
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm_f)  # HxWx4 floats 0..1
    rgb = (rgba[:, :, :3] * 255).astype('uint8')
    return rgb

def main(args):
    # Prepare pipeline (device=-1 for CPU; device=0,1,... for CUDA)
    device = args.device
    if device is None:
        device = -1
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

    # Open video capture (robust: try platform-specific backends if needed)
    cap = None
    if args.camera is not None:
        # Try default open first
        cap = cv2.VideoCapture(int(args.camera))
        # On macOS OpenCV may require AVFoundation backend explicitly
        if not cap.isOpened():
            try:
                import platform
                if platform.system() == 'Darwin' and hasattr(cv2, 'CAP_AVFOUNDATION'):
                    cap = cv2.VideoCapture(int(args.camera), cv2.CAP_AVFOUNDATION)
            except Exception:
                pass
    else:
        cap = cv2.VideoCapture(str(args.input))

    if cap is None or not cap.isOpened():
        # Helpful diagnostics for common macOS issues
        import platform
        print("Failed to open video input.")
        if args.camera is not None:
            print("- Tried to open camera index:", args.camera)
            if platform.system() == 'Darwin':
                print("- On macOS, ensure the Terminal/Python process has Camera permission:")
                print("  System Settings -> Privacy & Security -> Camera -> enable for your terminal app (Terminal/iTerm).")
                print("  If the terminal isn't listed, run a small Python test to trigger the permission prompt:")
                print('    python -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION); print(cap.isOpened()); cap.release()"')
                print("  You can also list devices using ffmpeg (install via Homebrew):")
                print('    ffmpeg -f avfoundation -list_devices true -i ""')
        else:
            print("- Tried to open input file:", args.input)
        raise RuntimeError("Failed to open input")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {w}x{h} @ {fps} FPS")

    # Output writer (same size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, fps / max(1, args.skip), (w, h))

    frame_idx = 0
    processed = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.skip > 1 and (frame_idx % args.skip != 0):
                frame_idx += 1
                continue

            # Convert BGR (cv2) -> RGB (PIL)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Run depth pipeline
            out_dict = pipe(pil)
            d = out_dict["depth"]

            # Convert to numpy and normalize
            arr = np.array(d)
            # If arr is HxWx3, collapse to luminance
            if arr.ndim == 3:
                arr = arr[..., 0]
            # Normalize to 0..1 float
            arrf = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

            if args.mode == 'color':
                rgb_vis = apply_colormap((arrf * 255).astype('uint8'), cmap_name=args.colormap)
                out_frame = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
            else:
                gray = normalize_to_uint8(arr)
                out_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            out.write(out_frame)
            processed += 1
            frame_idx += 1

            if processed % 10 == 0:
                elapsed = time.time() - t0
                print(f"Processed {processed} frames, avg {elapsed/processed:.3f}s/frame")

    finally:
        cap.release()
        out.release()
    total_t = time.time() - t0
    print(f"Done. Processed {processed} frames in {total_t:.1f}s ({total_t/max(1,processed):.3f}s/frame). Output: {args.output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, help='Input video file')
    p.add_argument('--camera', type=int, help='Camera index (use instead of --input)')
    p.add_argument('--output', type=Path, required=True, help='Output video file (mp4 recommended)')
    p.add_argument('--skip', type=int, default=1, help='Process every Nth frame (default 1)')
    p.add_argument('--mode', choices=['color', 'gray'], default='color', help='Output color or grayscale')
    p.add_argument('--colormap', default='plasma', help='matplotlib colormap name to use (plasma, viridis, turbo, etc.)')
    p.add_argument('--device', type=int, default=None, help='CUDA device index, e.g. 0; leave unset for CPU')
    args = p.parse_args()

    if args.camera is None and args.input is None:
        p.error('Either --input or --camera must be provided')

    main(args)