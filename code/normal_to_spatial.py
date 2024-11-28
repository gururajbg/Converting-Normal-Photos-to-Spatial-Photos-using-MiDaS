import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


# Load MiDaS depth estimation model
def load_depth_model():
    model_type = "DPT_Large"  # MiDaS v3 - Large
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    return midas, transform, device


# Generate depth map from an image
def generate_depth_map(image_path, midas, transform, device):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return img, depth_map


# Generate stereo (left and right) images from RGB and depth map
def generate_stereo_images(rgb_image, depth_map, offset_factor=0.02):
    height, width, _ = rgb_image.shape
    left_image = np.zeros_like(rgb_image)
    right_image = np.zeros_like(rgb_image)

    for y in range(height):
        for x in range(width):
            depth_value = depth_map[y, x]  # Depth value in range 0-1
            offset = int(depth_value * offset_factor * width)

            if x - offset >= 0:
                left_image[y, x - offset] = rgb_image[y, x]
            if x + offset < width:
                right_image[y, x + offset] = rgb_image[y, x]

    return left_image, right_image


# Main function to load model, generate depth map, and create stereo images
def main(image_path):
    midas, transform, device = load_depth_model()
    rgb_image, depth_map = generate_depth_map(image_path, midas, transform, device)

    # Generate left and right stereo images
    left_image, right_image = generate_stereo_images(rgb_image, depth_map)

    # Display the original, left, and right images
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(rgb_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(left_image)
    plt.title("Left Eye View")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(depth_map)
    plt.title("Right Eye View")
    plt.axis('off')

    plt.show()


# Run the function
main("../data/desk.png")
