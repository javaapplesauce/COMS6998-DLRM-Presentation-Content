from transformers import pipeline
from PIL import Image
import requests

# small helper imports used when saving/visualizing outputs
import numpy as np

# load pipe
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
depth = pipe(image)["depth"]
arr = np.array(depth)

if arr is not None:
    mn, mx = float(arr.min()), float(arr.max())
    print(f"depth array shape={arr.shape}, min={mn}, max={mx}")

    # normalized 0..1
    norm = (arr - mn) / (mx - mn + 1e-8)

    # save grayscale visualization
    vis_gray = (norm * 255).astype('uint8')
    # if single-channel (H, W) -> save as grayscale; if HxWxC, convert to single-channel
    if vis_gray.ndim == 2:
        Image.fromarray(vis_gray).save("depth_gray.png")
    else:
        # collapse to first channel
        Image.fromarray(vis_gray[:, :, 0]).save("depth_gray.png")
    print("Saved depth_gray.png")

    # save warm-to-cold colored visualization (uses matplotlib colormap)
    try:
        import matplotlib.cm as cm
        # use a perceptually-uniform colormap that looks warm->cold; choose 'plasma' here
        cmap = cm.get_cmap('plasma')
        rgba = cmap(norm)
        rgb = (rgba[:, :, :3] * 255).astype('uint8')
        Image.fromarray(rgb).save("depth_color.png")
        print("Saved depth_color.png (plasma colormap)")
    except Exception as e:
        print("matplotlib not available - to save colored depth map run: pip install matplotlib")

import numpy as np
from PIL import Image

# Inspect
print("depth type:", type(depth))
try:
    arr = np.array(depth)
    print("depth shape:", arr.shape, "dtype:", arr.dtype)
except Exception as e:
    print("Couldn't convert depth to numpy array:", e)
    arr = None

# Save a normalized visualization if conversion succeeded
if arr is not None:
    # normalize to 0-255 for viewing
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    vis = (norm * 255).astype("uint8")
    # if single-channel, convert to grayscale PIL image
    if vis.ndim == 2:
        Image.fromarray(vis).save("depth_vis.png")
    else:
        # if it's HxWx3, save directly
        Image.fromarray(vis).save("depth_vis.png")
    print("Saved depth visualization to depth_vis.png")