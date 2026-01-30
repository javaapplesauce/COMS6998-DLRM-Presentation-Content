# COMS6998 DLRM Presentation Content

This repo contains demos used in the presentation, including small scripts for depth estimation using the "Depth-Anything" models and utilities to render/manipulate visualizations.

## setup


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Demos
- `demo/depth-anything.py`
    ```bash
    python demo/depth-anything.py --input test1.jpg --output depth_out.png --mode
    color
    ```
- `demo/depth-anything-video.py`
  ```bash
  python demo/depth-anything-video.py --input test1.MP4 --output depth_out.mp4 --mode color
  ```
  ```bash
  python demo/depth-anything-video.py --camera 0 --output depth_cam.mp4 --mode color
  ```
  ```bash
  python demo/depth-anything-video.py --input test1.MP4 --output depth_out.mp4 --mode color --downscale 2.0 --skip 2
  ```
- `demo/depth-anything-lightning.py`
  ```bash
  python3 demo/depth-anything-lightning.py --input test1.MP4 --output depth_out.mp4 --batch-size 8 --num-workers 4 --downscale 2.0 --device cpu
  ```