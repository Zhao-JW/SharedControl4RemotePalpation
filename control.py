import numpy as np
import open3d as o3d

def get_target_position(keys, control_input, vel):
        # velocity set limit set in configurations
        lin_vel, rot_vel, press_vel = vel       
        target_tcp = control_input.copy()

        # can add hard limit to robot pose here
        # ignoring rotation control at the moment
        # the input control_input is the current tcp pose, return the target tcp position
        # ------------   X -----------------------
        if "+X" in keys:
            target_tcp[0] += lin_vel

        if "-X" in keys:
            target_tcp[0] -= lin_vel

        # ------------   Y -----------------------
        if "+Y" in keys:
            target_tcp[1] += lin_vel

        if "-Y" in keys:
            target_tcp[1] -= lin_vel

        # # ------------   PRESS ----------------------- press now will be different
        # if "PRESS" in keys:
        #     target_tcp[2] -= press_vel

        # if "LIFT" in keys:
        #     target_tcp[2] += press_vel
        return target_tcp[:2]

def new_control_input(new_pos, filename):
    # new_pos: the xy(z) of the new pose after moving the robot
    data = np.load("calibData.npy")
    theta, tx, ty, tz = data
    x, y = new_pos
    pcd = generate_point_cloud(filename, visualize=False)
    convert_point_cloud(pcd, data)

    # check the points are not too sparse





def generate_point_cloud(filename, visualize = True):
    color_raw = o3d.io.read_image(filename+"_rgb.png")
    depth_raw = o3d.io.read_image(filename+"_d.png")
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,depth_raw, convert_rgb_to_intensity= False)
    intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_parameters/intrinsic.json")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
        # o3d.camera.PinholeCameraIntrinsic(
        #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def convert_point_cloud(pcd, data):
    theta, tx, ty, tz = data
    points = np.asarray(pcd.points)
    for i, p in enumerate(points):
        newpos = np.zeros(3)
        newpos[0] = p[0]*np.cos(theta) - p[1]*np.sin(theta) + tx
        newpos[1] = p[0]*np.sin(theta) - p[1]*np.cos(theta) + ty
        newpos[2] = p[2] + tz
        points[i] = newpos
    pcd.points = o3d.utility.Vector3dVector(points)

    # need to check if the parameters are good
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.5, max_nn=30))
    # return pcd

