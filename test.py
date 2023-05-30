from turtle import color
import pyrealsense2
import open3d as o3d
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans



print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(".\data\\1.ply")
# pcd = o3d.io.read_point_cloud(".\data\\flatPhantom\\00\\00_1458_3d_mesh.ply")
o3d.visualization.draw_geometries([pcd])


R,G,B = np.asarray(pcd.colors)[:,0],np.asarray(pcd.colors)[:,1],np.asarray(pcd.colors)[:,2]
# lightness = []
# for point in np.asarray(pcd.colors):
#     result = colorsys.rgb_to_hls(*point)
#     lightness.append(result[2])
# greyscale = lightness

points = []
greyscale = 0.3*R + 0.59*G + 0.11*B
thresh = 0.6*max(greyscale) + 0.4*min(greyscale)
print(max(greyscale), min(greyscale))
for i in range(len(greyscale)):
    if greyscale[i] > thresh:
        points.append(np.asarray(pcd.points)[i])
pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])
downpcd = pcd.voxel_down_sample(voxel_size=0.01)
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.5, max_nn=30))
o3d.visualization.draw_geometries([downpcd])

# # remove background via 6 degree K means
# rowsum_c = np.asarray(pcd.colors).sum(axis = 1)
# color_normalize = np.asarray(pcd.colors)/rowsum_c[:,np.newaxis]
# rowsum_p = np.asarray(pcd.points).sum(axis = 1)
# point_normalize = np.asarray(pcd.points)/rowsum_p[:,np.newaxis]
# rgbxyz = np.concatenate((color_normalize,point_normalize),axis=1)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(color_normalize)
# inlier_cloud = pcd.select_by_index(kmeans.labels_)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = pcd.select_by_index(kmeans.labels_, invert=True)
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

#convert to shperical coord and plot
normals = np.asarray(downpcd.normals)
theta = np.arctan(normals[:,1]/normals[:,0])
phi = np.arccos(normals[:,2])
for i in range(len(theta)):
#     if theta[i] < 0:
#         theta[i] += np.pi
    # print(normals[i,2], np.arccos(normals[i,2]))
    if phi[i] > np.pi/2:
        phi[i] = np.pi - phi[i]
# theta = np.abs(theta)
# phi = np.abs(phi-np.pi/2)
plt.scatter(theta,phi)
ave = [np.average(theta),np.average(phi)]
plt.scatter(ave[0],ave[1],color='red')
plt.show()
print(ave[1]/np.pi*180)


# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                          ransac_n=3,
#                                          num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])

# print("Downsample the point cloud with a voxel of 0.05")
# downpcd = pcd.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([downpcd])
# downpcd = pcd

# print("Recompute the normal of the downsampled point cloud")
# downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#     radius=0.01, max_nn=30))
# o3d.visualization.draw_geometries([downpcd])

# print("Print a normal vector of the 0th point")
# print(downpcd.normals[0])
# print("Print the normal vectors of the first 10 points")
# print(np.asarray(downpcd.normals)[:10, :])
# print("")

# data = np.asarray(downpcd.normals)
# filler = np.zeros(data.shape)
# combined = np.concatenate([filler,data],1)
# X,Y,Z,U,V,W = zip(*combined)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(X,Y,Z,U,V,W)
# ax.set_xlim([-1, 0.5])
# ax.set_ylim([-1, 1.5])
# # ax.set_zlim([-1, 8])
# plt.show()
