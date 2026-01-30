# demo/depth_lightning_infer.py
import os
from pathlib import Path
import time
from typing import List

import cv2
import numpy as np
from PIL import Image
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoConfig, AutoImageProcessor

def pil_collate(batch):
    return batch

class FrameDataset(Dataset):
    def __init__(self, frames: List[Image.Image], transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        if self.transform:
            img = self.transform(img)
        return img

class DepthPredictor(L.LightningModule):
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device='cpu'):
        super().__init__()
        
        # load processor + model manually for inference
        from transformers import pipeline
        self.model_name = model_name
        self.device_ = device
        self.processor = None
        self.model = None

    def setup(self, stage=None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.device_).eval()

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # batch is a list/tensor of PIL images
        inputs = self.processor(images=batch, return_tensors="pt").to(self.device_)
        out = self.model(**inputs)
        depth = out.predicted_depth  # (B, 1, H, W)
        depth_np = depth.squeeze(1).cpu().numpy()  # (B, H, W)
        return depth_np

# read frames from video
def read_video_frames(video_path, max_frames=None, downscale=None):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if max_frames and i > max_frames:
            break
        if downscale:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w/downscale), int(h/downscale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        frames.append(pil)
    cap.release()
    return frames

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='depth_out.mp4')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--downscale', type=float, default=2.0, help='Downscale factor for faster inference (2 -> half width/height)')
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    # CPU thread tuning
    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    import torch
    torch.set_num_threads(4)

    frames = read_video_frames(args.input, max_frames=args.max_frames, downscale=args.downscale)
    ds = FrameDataset(frames)
    # Use a collate function that returns the list of PIL Images unchanged so the processor
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pil_collate,
        persistent_workers=(args.num_workers > 0),
    )

    model = DepthPredictor(device=args.device)
    trainer = L.Trainer(accelerator="cpu", devices=1, precision=32) 

    start = time.time()
    preds = trainer.predict(model, dataloaders=dl)
    import numpy as np
    depths = np.concatenate(preds, axis=0)

    print("Pred shapes:", depths.shape, "time:", time.time()-start)

    # Save a quick video of colorized depths using matplotlib colormap
    import matplotlib.cm as cm
    from PIL import Image
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = depths.shape[1], depths.shape[2]
    out = cv2.VideoWriter(args.output, fourcc, 25.0, (w, h))
    cmap = cm.get_cmap('plasma')
    for d in depths:
        df = (d - d.min()) / (d.max() - d.min() + 1e-8)
        rgba = cmap(df)
        rgb = (rgba[:, :, :3] * 255).astype('uint8')
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()
    print("Saved", args.output)

if __name__ == '__main__':
    main()