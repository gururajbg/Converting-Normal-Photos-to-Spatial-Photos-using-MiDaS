import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time


def depth_to_pointcloud(depth_map, rgb_image, focal_length=1000):
    height, width = depth_map.shape
    x_grid, y_grid = np.meshgrid(range(width), range(height))
    Z = depth_map.reshape(-1)
    valid_points = Z > np.percentile(Z, 1)
    Z = Z[valid_points]
    X = ((x_grid.reshape(-1)[valid_points] - width / 2) * Z) / focal_length
    Y = ((y_grid.reshape(-1)[valid_points] - height / 2) * Z) / focal_length
    points = np.stack([X, Y, Z], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb_image is not None:
        colors = rgb_image.reshape(-1, 3)[valid_points] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd

def visualize_point_cloud(pcd):
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Get render options and modify them
    opt = vis.get_render_option()
    opt.point_size = 1.0  # Adjust point size
    opt.background_color = np.asarray([0, 0, 0])  # Black background

    # Get camera control
    vc = vis.get_view_control()

    # Set default viewpoint
    vc.set_zoom(0.8)
    vc.set_front([0, 0, -1])
    vc.set_up([0, -1, 0])

    print("Controls:")
    print("- Left click + drag: Rotate")
    print("- Right click + drag: Pan")
    print("- Mouse wheel: Zoom")
    print("- '[' or ']': Change point size")

    # Run the visualization
    vis.run()
    vis.destroy_window()
def generate_depth_map_and_pointcloud(image_path):
    start_time = time.time()
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map_viz = (depth_map * 255).astype(np.uint8)
    pcd = depth_to_pointcloud(depth_map, img)

    output_ply_path = "static/point_cloud.ply"
    o3d.io.write_point_cloud(output_ply_path, pcd)

    depth_colored = cv2.applyColorMap(depth_map_viz, cv2.COLORMAP_INFERNO)
    cv2.imwrite("static/depth_map.png", depth_colored)
    visualize_point_cloud(pcd)
    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"FPS: {1 / elapsed_time:.2f}")
    return depth_map, pcd
