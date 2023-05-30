import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import copy
from sklearn.cluster import KMeans
import scipy.optimize


def thresh_normal(pcd, greyscale, alpha, refcount):
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
    k_means = KMeans(init='k-means++', n_clusters=refcount, n_init=10)
    k_means.fit(points)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    return k_means_cluster_centers


def capture():
    # filename = "data\\calib"
    filename = "data\\" + input("input save filename: ")
    # We want the points object to be apersistent so we can display the last cloud when a frame drops
    points = rs.points()

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    align = rs.align(rs.stream.color)
    pipe = rs.pipeline()
    config = rs.config()
    # Enable depth stream
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)  #, 640, 480, rs.format.bgr8, 30


    # Start streaming with chosen configuration
    profile = pipe.start(config)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipe.wait_for_frames()


    try:
        # Wait for the next set of frames from the camera
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            align = rs.align(rs.stream.color)
            frameset = align.process(frames)
            
            
            color = np.asanyarray(color_frame.get_data())
            # fix color display
            b = color[:,:,0]
            g = color[:,:,1]
            r = color[:,:,2]
            color = np.concatenate((r[:,:,np.newaxis],g[:,:,np.newaxis],b[:,:,np.newaxis]),axis=2)

            plt.rcParams["axes.grid"] = False
            plt.rcParams['figure.figsize'] = [12, 6]
            # plt.imshow(color)

            depth_frame = frameset.get_depth_frame()        #access with np.asarray(depth_frame.data)
            colorizer = rs.colorizer()
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            images = np.hstack((color, colorized_depth))
            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)
            cv2.resizeWindow('Align Example', 2560, 720)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        cv2.imwrite(filename+"_rgb.png",color)
        cv2.imwrite(filename+"_d.png", np.asarray(depth_frame.data))
        print("image saved to "+filename)
        
    finally:
        pipe.stop()

def generate_point_cloud(filename, visualize = False):
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

def segment():
    print("Load a RGBD data")
    filename = "data/" + input("input file to open: ")
    offsetpos = input("input robot pose when capturing image:").split(" ")
    offsetpos = [float(i) for i in offsetpos]
    pcd = generate_point_cloud(filename)
    points = np.asarray(pcd.points)
    # remove background via greyscale thresholding
    R,G,B = np.asarray(pcd.colors)[:,0],np.asarray(pcd.colors)[:,1],np.asarray(pcd.colors)[:,2]

    pcd1 = copy.copy(pcd)
    # thresh_normal(pcd1, R-B-G, 0.6)
    refcount = 4
    centers = thresh_normal(pcd1, R+B+G, 0, refcount)
    a = sorted(centers,key=lambda x:x[0])
    centers = sorted(a[:2], key=lambda x:x[1]) + sorted(a[2:], key=lambda x:x[1])
    print("marker centers: ", centers)
    robot = input("input robot poses:").split(" ")
    robot = [float(i) for i in robot]
    robot = np.reshape(robot, (np.size(robot)//3, 3))
    
    
    # assume only z translation no rotation, and z plane parallel to robot z plane
    res = scipy.optimize.minimize(transformation, [0.3, 0, 0, 0], (robot, centers, offsetpos), method = 'Nelder-Mead')
    print(res)
    # calibration not nessesary everytime if the starting position is fixed, save the result in config file
    np.save("calibData", res.x)
    return res.x, robot, centers

def transformation(x, rpos, cpos, offsetpos):
    theta, tx, ty, tz = x
    sum = 0
    rpos_offset = - 0.045 # the tactile sensor was touching the table, not true tcp of the robot
    # offsetpos: the coord of robot when capturing the image
    for count, center in enumerate(rpos):
        center = cpos[count]
        diffx = center[0]*np.cos(theta) - center[1]*np.sin(theta) + tx + offsetpos[0] - rpos[count][0]
        diffy = center[0]*np.sin(theta) + center[1]*np.cos(theta) + ty + offsetpos[1]- rpos[count][1]
        diffz = center[2] + tz + offsetpos[2] - (rpos[count][2] + rpos_offset)
        sum += np.linalg.norm(np.array([diffx, diffy, diffz]))
    return sum

def resultCheck(x):
    # print("remainder: ", transformation(res, robot, centers))
    testpos = input("input test point camera x y z position: ").split(' ')
    offsetpos = input("input robot pose when capturing image:").split(" ")
    rpos_offset = - 0.045
    offsetpos = [float(i) for i in offsetpos]
    center = [float(i) for i in testpos]
    theta, tx, ty, tz = x
    xR = center[0]*np.cos(theta) - center[1]*np.sin(theta) + tx + offsetpos[0]
    yR = center[0]*np.sin(theta) + center[1]*np.cos(theta) + ty + offsetpos[1]
    zR = center[2] + tz + offsetpos[2] - rpos_offset
    print("robot position: ", xR, yR, zR)
    
    
# capture()
res, robot, centers = segment()
# res = np.array([4.72545143, -3.74754849,  2.36866799, -0.47634628])
# res = np.array([-1.56904745,  0.09569248,  0.29127733,  0.02871786])
resultCheck(res)