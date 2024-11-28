import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'


def apply_perspective_shift(image, depth_map, shift_scale=50):
    """
    Applies a perspective shift to the image based on the depth map.

    Parameters:
        image (numpy array): Original 2D image.
        depth_map (numpy array): Depth map corresponding to the image.
        shift_scale (int): Scale for pixel shifting based on depth values.

    Returns:
        numpy array: Perspective-corrected image.
    """
    h, w = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Scale depth values for pixel shift
    depth_scaled = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_shift = (depth_scaled - 0.5) * shift_scale  # Center around zero shift

    # Apply shift based on depth
    x_shifted = x_coords + depth_shift
    y_shifted = y_coords  # Vertical perspective adjustment can be added similarly

    # Ensure coordinates are within bounds
    x_shifted = np.clip(x_shifted, 0, w - 1).astype(np.float32)
    y_shifted = np.clip(y_shifted, 0, h - 1).astype(np.float32)

    # Remap image based on shifted coordinates
    corrected_image = cv2.remap(image, x_shifted, y_shifted, interpolation=cv2.INTER_LINEAR)
    return corrected_image

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('../data/desk.png')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
corrected_img = apply_perspective_shift(raw_img, depth)

# Display the depth map
plt.imshow(depth_normalized, cmap='plasma')
plt.title("Estimated Depth Map")
plt.colorbar()
plt.show()
plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
plt.title("Perspective-Corrected Image")
plt.axis('off')
plt.show()
print(123)