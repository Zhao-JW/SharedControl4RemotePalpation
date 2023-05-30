import open3d as o3d
import numpy as np
import scipy.interpolate

def load():
    print("Load a RGBD data")
    # pcd = o3d.io.read_point_cloud(".\data\\flatPhantom\\45\\45_28081_3d_mesh.ply")
    # filename = "data/" + input("input file to open: ")
    filename = "data/phantom1"
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
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def thresh_normal(pcd):
    points = []
    colors = []
    # print(max(greyscale), min(greyscale))
    
    #down sample the point cloud
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.5, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    downpoints = np.asarray(downpcd.points)
    sorting = np.lexsort((downpoints[:,1],downpoints[:,0]))
    sortpts = downpoints[sorting,:]

    return downpcd, sortpts

def makeGrid(sorteddpts):
    x = np.linspace(0.041,0.048)
    y = np.linspace(-0.04,0)
    xx,yy = np.meshgrid(x,y)
    fxx = scipy.interpolate.griddata(sorteddpts[:,:2],sorteddpts[:,2],(xx.ravel(), yy.ravel()),'linear')
    xSpline = scipy.interpolate.RectBivariateSpline(x,y,fxx)
    return xSpline

def evaluateGrid(xSpline, position):
    pose = xSpline.ev(*position)
    return pose

if __name__ == "__main__":
    pcd = load()
    pcd,sortpts = thresh_normal(pcd)
    xSpline = makeGrid(sortpts)
    position = [0.042,-0.042]
    evaluateGrid(xSpline,position)