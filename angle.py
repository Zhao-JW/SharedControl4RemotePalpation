import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy

def thresh_normal(pcd, greyscale, alpha):
    points = []
    colors = []
    thresh = alpha*max(greyscale) + (1-alpha)*min(greyscale)
    print(thresh)
    # print(max(greyscale), min(greyscale))
    for i in range(len(greyscale)):
        if greyscale[i] > thresh:
            points.append(np.asarray(pcd.points)[i])
            colors.append(np.asarray(pcd.colors)[i])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    
    #down sample the point cloud
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.5, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    downpoints = np.asarray(downpcd.points)
    sorteddpts = np.lexsort((downpoints[:,1],downpoints[:,0]))

    #find normal, convert to shperical coord and plot
    normals = np.asarray(downpcd.normals)
    theta = np.arctan(normals[:,1]/normals[:,0])
    theta = theta/np.pi*180
    phi = np.arccos(np.abs(normals[:,2])/1)     # assume normal always point outwards
    phi = phi/np.pi*180
    for i in range(len(theta)):
        if theta[i] < 0:
            # theta[i] += np.pi 
            theta[i] += 180
            pass
    # return None,None
    points = np.concatenate((downpoints,np.vstack((theta,phi)).T), axis=1)
    return theta, phi

def custom_draw_geometry_with_rotation(pcd):
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            return False
            
        o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)


print("Load a RGBD data")
# pcd = o3d.io.read_point_cloud(".\data\\flatPhantom\\45\\45_28081_3d_mesh.ply")
filename = "data/" + input("input file to open: ")
color_raw = o3d.io.read_image(filename+"_rgb.png")
depth_raw = o3d.io.read_image(filename+"_d.png")
# depth_raw.data /= np.log(np.asarray(depth_raw))
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,depth_raw, convert_rgb_to_intensity= False)


## image plotting 2D
# print(rgbd_image)
# plt.subplot(1, 2, 1)
# plt.title('grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()
intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_parameters/intrinsic.json")
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, intrinsic)
    # o3d.camera.PinholeCameraIntrinsic(
    #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
# custom_draw_geometry_with_rotation(pcd)
points = np.asarray(pcd.points)


# remove background via greyscale thresholding
R,G,B = np.asarray(pcd.colors)[:,0],np.asarray(pcd.colors)[:,1],np.asarray(pcd.colors)[:,2]

pcd1 = copy.copy(pcd)
# thresh_normal(pcd1, R-B-G, 0.6)
theta_ref, phi_ref = thresh_normal(pcd1, R-B-G, 0.6)
tilt_ref = np.average(phi_ref)
if np.average(theta_ref) > np.pi/2:
    tilt_ref = - tilt_ref
greyscale = 0.3*R + 0.59*G + 0.11*B
theta, phi = thresh_normal(pcd, greyscale, 0.6)
# thresh_normal(pcd, greyscale, 0.6)
# phi = phi + 6.19

plt.scatter(theta,phi)
ave = [np.average(theta),np.average(phi)]
plt.scatter(ave[0],ave[1],color='red')
if ave[0] > np.pi/2:
    tilt = - ave[1]
else:
    tilt = ave[1]
print("estimated angle: ", ave[1])
print("after camera tilt correction: ", (tilt+np.pi-tilt_ref))
# plt.xlim([-np.pi/2,np.pi/2])
plt.xlim([0,180])
# plt.ylim([0, np.pi/2])
plt.ylim([0, 90])
plt.xlabel("theta/rotation about z axis")
plt.ylabel("phi/tile angle (assumed 0~90 degree)")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(100, 85, 'ground truth: 45 \nestimated: '+str(round(ave[1],2))+' \ntilt correction: '+str(round(ave[1]+6.19,2)), fontsize=14,
        verticalalignment='top', bbox=props)
# plt.text(-5, 60, 'ground truth: 0 \n estimated: 6.19 \n camera tile correction: na', fontsize = 16)
plt.show()
# theta might not be in a blob but it doesn't matter as much

# # method2: find largest plane of the threshholded result/find plane once, remove and find the second largest one
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                          ransac_n=3,
#                                          num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# print("normal: ",a,b,c," theta: ",np.arctan(b/a)," phi: ",np.arccos(c)/np.pi*180)
# inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


