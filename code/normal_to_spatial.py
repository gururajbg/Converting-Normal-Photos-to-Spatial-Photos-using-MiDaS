import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def depth_to_pointcloud(depth_map, rgb_image, focal_length=1000):
    height, width = depth_map.shape

    # Create grid of image coordinates
    x_grid, y_grid = np.meshgrid(range(width), range(height))

    # Calculate 3D coordinates
    Z = depth_map.reshape(-1)
    X = ((x_grid.reshape(-1) - width / 2) * Z) / focal_length
    Y = ((y_grid.reshape(-1) - height / 2) * Z) / focal_length

    # Stack coordinates
    points = np.stack([X, Y, Z], axis=1)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add colors if RGB image is provided
    if rgb_image is not None:
        colors = rgb_image.reshape(-1, 3) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def generate_depth_map_and_pointcloud(image_path):
    # Load model
    model_type = "DPT_Large"  # MiDaS v3 - Large

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Generate depth map
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth map
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map_viz = (depth_map * 255).astype(np.uint8)

    # Create point cloud
    pcd = depth_to_pointcloud(depth_map, img)

    # Apply colormap for visualization
    depth_colored = cv2.applyColorMap(depth_map_viz, cv2.COLORMAP_INFERNO)

    # Display results
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
    plt.title('Depth Map')
    plt.axis('off')

    plt.show()

    # Visualize point cloud
    o3d.visualization.draw_geometries([pcd])

    return depth_map, pcd


# Example usage
image_path = "../data/test.jpg"  # Replace with your image path
depth_map, point_cloud = generate_depth_map_and_pointcloud(image_path)