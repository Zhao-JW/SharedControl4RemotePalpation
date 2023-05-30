import pyrealsense2 as rs
import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
import copy
import scipy
import os.path
import scipy.interpolate

class SHAREDCONTROl:
    def __init__(self):
        self.filename = None
        self.pcd = None     # downsampled segmented point cloud with estimated normal
        self.points = None  # sorted 6D data points

    def genMap(self, filename=None):
        # capture data and save, then load and estimate normal and updat pcd&points value of this object, called at the start of experiment
        # input file name, popup window wih rgb and depth rendering, press Q to exit and capture
        if not filename:
            filename = input("input file name: ")
        self.filename = "data\\" + filename

        if not (os.path.isfile(self.filename + "_rgb.png") and os.path.isfile(self.filename + "_d.png")):
            self.captureRGBD(filename)
        self.estimateNormal()
        self.interpolate_prep()

    def loadMap(self, filename):
        self.filename = "data\\" + filename +"_points.npy"
        self.points = np.load(self.filename)
        self.interpolate_prep()


    def genControl(self, control_input, keys, vel):
        target_position = self.get_target_position(keys, control_input,vel)
        control_input_new = self.getControl(target_position, control_input)
        print("current pose: ", control_input, "target pose:", control_input_new)
        # return [*target_position, *control_input[3:]]
        return control_input_new
        # return the converted control input to robot knowing current pos
    
    def interpolate_prep(self):
        # tree search method
        # self.tree = scipy.spatial.KDTree(self.points[:, :2])

        # grid method
        # xx = np.linspace(np.min(self.points[:, 0]), np.max(self.points[:, 0]))
        # yy = np.linspace(np.min(self.points[:,1]), np.max(self.points[:,1]))
        xx = np.linspace(-0.55, -0.2)
        yy = np.linspace(-0.2, 0.15)
        n = xx.size
        xx_grid, yy_grid = np.meshgrid(xx, yy)
        fzza = scipy.interpolate.griddata((self.points[:, 0], self.points[:, 1]), self.points[:, 2],
                                         (xx_grid.ravel(), yy_grid.ravel()), fill_value=0, method='cubic')
        fxx = scipy.interpolate.griddata((self.points[:, 0], self.points[:, 1]), self.points[:, 3],
                                         (xx_grid.ravel(), yy_grid.ravel()), fill_value=0, method='cubic')
        fyy = scipy.interpolate.griddata((self.points[:, 0], self.points[:, 1]), self.points[:, 4],
                                         (xx_grid.ravel(), yy_grid.ravel()), fill_value=0, method='cubic')
        fzz = scipy.interpolate.griddata((self.points[:, 0], self.points[:, 1]), self.points[:, 5],
                                         (xx_grid.ravel(), yy_grid.ravel()), fill_value=1, method='cubic')

        # normalize the normal interpolated
        for i in range(xx_grid.size):
            norm = np.linalg.norm([fxx[i],fyy[i], fzz[i]])
            fxx[i] /= norm
            fyy[i] /= norm
            fzz[i] /= norm

        # fza = scipy.interpolate.RectBivariateSpline(xx, yy, fzza.reshape(n, n))
        # fx = scipy.interpolate.RectBivariateSpline(xx, yy, fxx.reshape(n, n))
        # fy = scipy.interpolate.RectBivariateSpline(xx, yy, fyy.reshape(n, n))
        # fz = scipy.interpolate.RectBivariateSpline(xx, yy, fzz.reshape(n, n))
        self.fx = fxx
        self.fy = fyy
        self.fz = fzz
        self.fza = fzza
        self.grid = (xx, yy)


    @property

    def captureRGBD(self, filename = None):
        # this function captures the rgbd data and saves it as two files

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
            cv2.imwrite(self.filename+"_rgb.png",color)
            cv2.imwrite(self.filename+"_d.png", np.asarray(depth_frame.data))
            print("image saved to "+self.filename)

        finally:
            pipe.stop()

    def convertMap(self, pcd):
        # pre-calculated calibration parameters, theta in rad, txtytz in meter
        # theta, tx, ty, tz = [-1.57606377,  0.07090451,  0.19130725, -0.01829447]
        theta, tx, ty, tz = [-1.57606377,  -0.05, 0, 0]
        # calib2 set result

        #position of the robot when taking that picture, need to be the same as when getting transformation
        offsetpos = [-0.349, -0.047, 0.416]    # rx, ry, rz = 1.189 2.933 -0.021

        #distance between the tactile sensor and TCP of the robot
        rpos_offset = -0.045

        points = np.asarray(pcd.points)
        for i, point in enumerate(points):
            px = point[0] * np.cos(theta) - point[1] * np.sin(theta) + tx + offsetpos[0]
            py = point[0] * np.sin(theta) + point[1] * np.cos(theta) + ty + offsetpos[1]
            pz = point[2] + tz + offsetpos[2] - rpos_offset
            points[i] = [px,py,pz]
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def estimateNormal(self, convert=True):
        # get the calibrated theta phi of the data at each point
        if os.path.isfile(self.filename+"_points.npy"):
            self.points = np.load(self.filename+"_points.npy")

        if self.filename:       # if filename not None
            # load data
            print("Load a RGBD data")
            color_raw = o3d.io.read_image(self.filename+"_rgb.png")
            depth_raw = o3d.io.read_image(self.filename+"_d.png")
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,depth_raw, convert_rgb_to_intensity= False)

            # convert the RGBD data to point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # convert the pcd map from camera frame to robot frame
            if convert:
                # call convertmap function
                pcd = self.convertMap(pcd)

            # remove background via greyscale thresholding
            R,G,B = np.asarray(pcd.colors)[:,0],np.asarray(pcd.colors)[:,1],np.asarray(pcd.colors)[:,2]

            # pcd1 = copy.copy(pcd)
            greyscale = 0.3*R + 0.59*G + 0.11*B
            normals, self.pcd = self.thresh_normal(pcd, greyscale=None)

            # store the points as sorted 5D data (xyz theta phi)
            points = np.asarray(self.pcd.points)
            self.points = np.concatenate((points,normals),axis = 1)
            # points = self.points.copy()
            idx = np.lexsort((self.points[:, 1], self.points[:, 0]))
            sortpts = self.points[idx]
            self.points = sortpts
            np.save(self.filename+"_points.npy", sortpts)
            # print("workspace range found: ", np.max(points[:,0]), np.min(points[:,0]),np.max(points[:,1]), np.min(points[:,1]))
            # self.points = np.concatenate((points,np.vstack((theta,phi)).T), axis=1)
            # self.tree = scipy.spatial.KDTree(sortpts[:, :2])
        else:
            print("need to capture data first")
        
    def thresh_normal(self, pcd, greyscale=None, alpha=0.6):
        # thresholding the given pcd data using the greyscale input and estimate the normals
        # theta[0,180], phi[0,90]: array of angles in degree wrt picture
        points = []
        colors = []
        # if not greyscale:
        #     thresh = alpha*max(greyscale) + (1-alpha)*min(greyscale)
        #     print("threshold: ",thresh)
        #     # print(max(greyscale), min(greyscale))
        #     for i in range(len(greyscale)):
        #         if greyscale[i] > thresh:
        #             points.append(np.asarray(pcd.points)[i])
        #             colors.append(np.asarray(pcd.colors)[i])
        #     pcd.points = o3d.utility.Vector3dVector(points)
        #     pcd.colors = o3d.utility.Vector3dVector(colors)

        #down sample the point cloud
        downpcd = pcd.voxel_down_sample(voxel_size=0.01)
        # downpcd = pcd
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.7, max_nn=30))

        # remove outlier
        cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=2.0)
        inlier_cloud = cl.select_by_index(ind)


        #find normal, convert to shperical coord and plot
        normals = np.asarray(inlier_cloud.normals)

        # enforce normal point outwards
        for i, normal in enumerate(normals):
            if normal[2] < 0:
                normals[i] = -normal

        inlier_cloud.normals = o3d.utility.Vector3dVector(normals)
        # o3d.visualization.draw_geometries([inlier_cloud])

        return normals, inlier_cloud

    def getControl(self,target_position, control_input):

        # get the xyz rxryrz coord at given xy point
        # position for xy only, pose for 6D data
        # the parameter target position and current position are both in tcp

        # kd tree method
        # _, nearidx = self.tree.query(target_position[:2], k=20, distance_upper_bound=0.03)
        # idxs = []
        # for idx in nearidx:
        #     if idx < np.shape(self.points)[0]:
        #        idxs += [idx]
        # neartarget = self.points[idxs]
        # fxx = scipy.interpolate.griddata((neartarget[:,:2]), neartarget[:,3], target_position[:2])
        # fyy = scipy.interpolate.griddata((neartarget[:,:2]), neartarget[:,4], target_position[:2])
        # fzz = scipy.interpolate.griddata((neartarget[:,:2]), neartarget[:,5], target_position[:2])

        # interpolate method
        # xx = np.linspace(-0.55, -0.2)
        # yy = np.linspace(-0.2, 0.15)
        # n = xx.size
        # normal = [float(self.fx.ev(*target_position[:2])), float(self.fy.ev(*target_position[:2])), float(self.fz.ev(*target_position[:2]))]
        # directly query the map instead
        xx, yy = self.grid
        n = xx.size
        dist = [xx[1] -xx[0], yy[1] - yy[0]]
        i = int((target_position[0] - xx[0])//dist[0])
        j = int((target_position[1] - yy[0])//dist[1])
        if i >= n-5 or i < 5 or j >= n-5 or j < 5:
            normal = [0,0,1]
        else:
            idx = int(i + j*n)
            # normal = [self.fx[idx], self.fy[idx], self.fz[idx]]

            # normal_x = np.interp(target_position[0], xx[i:i+2],self.fx[idx:idx+2])
            # normal_y = np.interp(target_position[1], yy[i:i+2],self.fy[idx:idx+2])

            coords = np.array([[xx[i],yy[j]], [xx[i+1],yy[j]],[xx[i],yy[j+1]],[xx[i+1],yy[j+1]]])
            idxs = [idx, idx+1, idx+n, idx+n+1]
            normal_x = scipy.interpolate.griddata(coords, self.fx[idxs],target_position[:2])
            normal_y = scipy.interpolate.griddata(coords, self.fy[idxs],target_position[:2])
            normal_z = scipy.interpolate.griddata(coords, self.fz[idxs],target_position[:2])
            normal = [normal_x[0], normal_y[0], normal_z[0]]
            

        # # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax.plot_surface(xx, yy, fxx.reshape(n,n), cmap='viridis', cstride=1, rstride=1)
        # # plt.show()


        # fxx = scipy.interpolate.griddata((sortpts[:,0], sortpts[:,1]), sortpts[:,3],target_position[:2])
        # fyy = scipy.interpolate.griddata((sortpts[:, 0], sortpts[:, 1]), sortpts[:, 4], target_position[:2])
        # fzz = scipy.interpolate.griddata((sortpts[:, 0], sortpts[:, 1]), sortpts[:, 5], target_position[:2])

        # normal = np.array([fxx[0], fyy[0], fzz[0]])
        target_pose = self.nomral2pose(np.array(normal))

        control_input = [*target_position, *target_pose]
        return control_input

    def nomral2pose(self, normal):
        # normal alone is not enough to determine tcp pose, enforce restriction: y' always in YZ plane
        z_new = -normal
        if z_new[1] == 0:
            if z_new[2] != 0:
                y_z = 0.
                y_y = 1.
            else:
                raise Exception
        else:
            k = -z_new[2]/z_new[1]
            np.sign(k)
            y_z = np.sign(k)*1/np.sqrt(1+k**2)
            y_y = np.sign(k)*k/np.sqrt(1+k**2)
        y_new = np.array([0., y_y, y_z])
        x_new = np.cross(y_new, z_new)
        R = np.vstack((x_new, y_new,z_new)).T
        rotation,_ = cv2.Rodrigues(R)

        return np.squeeze(rotation)

    def get_target_position(self, keys, control_input, vel):
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

        # ------------   PRESS -----------------------
        if "PRESS" in keys:
            target_tcp[2] -= press_vel

        if "LIFT" in keys:
            target_tcp[2] += press_vel
        return target_tcp[:3]



if __name__ == "__main__":
    a = SHAREDCONTROl()
    # pose = a.nomral2pose(np.array([np.sqrt(1-0.9**2-0.1**2),0.1,-0.9]))
    # print(pose)

    a.genMap("phantom2")
    # print(a.nomral2pose(np.array([0,0,1])))
    control_input = [-0.56, -0.09, 0.046, 0.181, 3.162, 0.056]
    keys = "+X"
    vel = [0.01, 0.01, 0.01]
    a.genControl(control_input, keys, vel)

    # xx = np.linspace(-0.6, -0.2)
    # yy = np.linspace(-0.3, 0.1)
    xx = np.linspace(-0.55, -0.2)
    yy = np.linspace(-0.2, 0.2)
    n = xx.size
    xx_grid, yy_grid = np.meshgrid(xx, yy)
    pose = np.zeros((2500, 3))
    for i in range(n**2):
        pose1 = np.array(a.getControl([xx_grid.ravel()[i], yy_grid.ravel()[i], 10], control_input)[3:])
        # pose1 = a.nomral2pose(np.array([fxx[i], fyy[i],fzz[i]]))
        start_pos = np.array([0,0,1])
        R,_ = cv2.Rodrigues(pose1)
        pos_new = np.matmul(start_pos, R)
        pose[i,:] = pos_new
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx_grid, yy_grid, -pose[:,2].reshape(n,n))#, cmap='viridis', cstride=1, rstride=1)
    plt.show()

    # normal = np.array([0.461, -0.022, 0.887])
    # normal = np.array([0.6, 0, 0.8])
    # pose1 = a.nomral2pose(normal)
    # start_pos = np.array([0,0,1])
    # R,_ = cv2.Rodrigues(pose1)
    # pose_new = np.matmul(start_pos, R)
    # print(pose1, pose_new)