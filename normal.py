import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy

def load():
    print("Load a RGBD data")
    # pcd = o3d.io.read_point_cloud(".\data\\flatPhantom\\45\\45_28081_3d_mesh.ply")
    # filename = "data/" + input("input file to open: ")
    filename = "data/phantom2"
    color_raw = o3d.io.read_image(filename+"_rgb.png")
    depth_raw = o3d.io.read_image(filename+"_d.png")
    # depth_raw.data /= np.log(np.asarray(depth_raw))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,depth_raw, convert_rgb_to_intensity= False)

    intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_parameters/intrinsic.json")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
        # o3d.camera.PinholeCameraIntrinsic(
        #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    return pcd

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
                                    #   zoom=0.3412,
                                    #   front=[0.4257, -0.2125, -0.8795],
                                    #   lookat=[2.6172, 2.0475, 1.532],
                                    #   up=[-0.0694, -0.9768, 0.2024])

def get_normal(pcd):
    points = []
    colors = []
    
    #down sample the point cloud
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.7, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=2.0)
    display_inlier_outlier(downpcd, ind)
    return downpcd

if __name__ == "__main__":
    pcd = load()
    get_normal(pcd)