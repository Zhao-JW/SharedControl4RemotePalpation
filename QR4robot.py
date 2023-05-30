import cv2 as cv
import numpy as np
import sys
import pyrealsense2 as rs
import matplotlib.pyplot as plt
# import rtde_io
# import rtde_receive

def read_camera_parameters(filepath = 'camera_parameters/intrinsic.dat'):
# need to change to actual camera parameters
    inf = open(filepath, 'r')

    cmtx = []
    dist = []

    #ignore first line
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    #ignore line that says "distortion"
    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    #cmtx = camera matrix, dist = distortion parameters
    return np.array(cmtx), np.array(dist)

def get_qr_coords(cmtx, dist, points):
    size = 30
    #Selected coordinate points for each corner of QR code.
    qr_edges = np.array([[0,0,0],
                         [0,size,0],
                         [size,size,0],
                         [size,0,0]], dtype = 'float32').reshape((4,1,3))
    # patternsize = [6, 8]  # [height,width]
    # objectPoints = np.zeros((patternsize[0] * patternsize[1], 3), np.float32)
    # objectPoints[:, :2] = np.mgrid[0:patternsize[0], 0:patternsize[1]].T.reshape(-1, 2)
    # qr_edges = objectPoints * 30

    #determine the orientation of QR code coordinate system with respect to camera coorindate system.
    ret, rvec, tvec = cv.solvePnP(qr_edges, points, cmtx, dist)
    # rvet tvec translate point in object frame to camera frame

    #Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
    unitv_points = np.array([[0,0,0], [size,0,0], [0,size,0], [0,0,size]], dtype = 'float32').reshape((4,1,3))
    if ret:
        points, jac = cv.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
        return points, rvec, tvec

    #return empty arrays if rotation and translation values not found
    else: return [], [], []


def show_axes(cmtx, dist, in_source):
    # cap = cv.VideoCapture(in_source)
    align = rs.align(rs.stream.color)
    pipe = rs.pipeline()
    config = rs.config()
    # Enable depth stream
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)  # , 640, 480, rs.format.bgr8, 30

    # Start streaming with chosen configuration
    profile = pipe.start(config)

    qr = cv.QRCodeDetector()
    results = []
    robotPose = []

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipe.wait_for_frames()
    try:
    # Wait for the next set of frames from the camera
        while True:
            # ret, img = cap.read()
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            align = rs.align(rs.stream.color)
            frameset = align.process(frames)

            color = np.asanyarray(color_frame.get_data())
            # fix color display
            b = color[:, :, 0]
            g = color[:, :, 1]
            r = color[:, :, 2]
            color = np.concatenate((r[:, :, np.newaxis], g[:, :, np.newaxis], b[:, :, np.newaxis]), axis=2)

            plt.rcParams["axes.grid"] = False
            plt.rcParams['figure.figsize'] = [12, 6]
            # plt.imshow(color)

            depth_frame = frameset.get_depth_frame()  # access with np.asarray(depth_frame.data)
            colorizer = rs.colorizer()
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            img = np.hstack((color, colorized_depth))

            # if ret == False: break
            ret_qr, points = qr.detect(color)
            print(points)
            # ret_qr, points = cv.findChessboardCorners(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (6, 8), None)

            if ret_qr:
                axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)

                #BGR color format
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,0,0)]

                #check axes points are projected to camera view.
                if len(axis_points) > 0:
                    axis_points = axis_points.reshape((4,2))
                    origin = (int(axis_points[0][0]),int(axis_points[0][1]) )

                    for p, c in zip(axis_points[1:], colors[:3]):
                        p = (int(p[0]), int(p[1]))

                        #Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases.
                        if origin[0] > 5*img.shape[1] or origin[1] > 5*img.shape[1]:break
                        if p[0] > 5*img.shape[1] or p[1] > 5*img.shape[1]:break

                        cv.line(img, origin, p, c, 5)

                # robot = rtde_receive_.getActualTCPPose()
                # robot[0] *= 1000
                # robot[1] *= 1000
                # robot[2] *= 1000
                # print("robot pose:", robot)
                print("oject2camera pose: ", np.squeeze(rvec), np.squeeze(tvec))
                results.append([rvec, tvec])
                # robotPose.append(robot)

            cv.imshow('frame', img)

            k = cv.waitKey(20)
            if k == 27: break #27 is ESC key.
            # if k == 32:

                # print("data recorded: ", rvec, tvec)
                # robot = input("input robot pose with space in between: ").split(' ')
                # robot = [float(data) for data in robot]

    
    # cap.release()
    # cv.destroyAllWindows()
    finally:
        pipe.stop()
    return results, robotPose

def handEyeCalibrate(refPose, robotPose):
    # refPose in (rvec,tvec), robotPose in [x,y,z,rx,ry,rz]
    R_target2cam = []
    t_target2cam = []
    for pose in refPose:
        R_target2cam.append(-1*cv.Rodrigues(np.squeeze(pose[0]))[0])
        t_target2cam.append(-1*np.squeeze(pose[1]).tolist())

    R_end2base = []
    t_end2base = []
    for pose in robotPose:
        # scale = 1 - 2*np.pi/np.sqrt(pose[3]**2+pose[4]**2+pose[5]**2)
        # scale = 1
        theta = np.linalg.norm(pose[3:])
        if theta>np.pi or theta<0:
            print(theta)
            break
        tvec = pose[:3]
        rvec = np.array([pose[5], pose[4],pose[3]])
        R_end2base.append(cv.Rodrigues(rvec)[0])
        t_end2base.append(tvec)
    
    R, t = cv.calibrateHandEye(
        R_gripper2base=np.array(R_end2base),
        t_gripper2base=np.array(t_end2base),
        R_target2cam=np.array(R_target2cam),
        t_target2cam=np.array(t_target2cam),
        method = 0
    )
    print("method0: ", R, t)
    R, t = cv.calibrateHandEye(
        R_gripper2base=np.array(R_end2base),
        t_gripper2base=np.array(t_end2base),
        R_target2cam=np.array(R_target2cam),
        t_target2cam=np.array(t_target2cam),
        method=1
    )
    print("method1: ", R, t)
    R, t = cv.calibrateHandEye(
        R_gripper2base=np.array(R_end2base),
        t_gripper2base=np.array(t_end2base),
        R_target2cam=np.array(R_target2cam),
        t_target2cam=np.array(t_target2cam),
        method=2
    )
    print("method2: ", R, t)
    R, t = cv.calibrateHandEye(
        R_gripper2base=np.array(R_end2base),
        t_gripper2base=np.array(t_end2base),
        R_target2cam=np.array(R_target2cam),
        t_target2cam=np.array(t_target2cam),
        method=3
    )
    print("method3: ", R, t)
    return R,t

if __name__ == '__main__':
    # rtde_io_ = rtde_io.RTDEIOInterface("169.254.250.80")
    # rtde_receive_ = rtde_receive.RTDEReceiveInterface("169.254.250.80")

    #read camera intrinsic parameters.
    cmtx, dist = read_camera_parameters()

    # input_source = 'QR_code_orientation_OpenCV-main/smedia/test.mp4'
    # if len(sys.argv) > 1:
    # input_source = int(sys.argv[1])
    input_source = 0
    print("press space to record, esc to quit")
    refPose, robotPose = show_axes(cmtx, dist, input_source)
    # print("all records: ",refPose, robotPose)
    # robotPose = [[0,0,0,0,0,0],
    #             [-0.15,0,0,0,0,0],
    #             [0,0,0.145,0,np.pi,0],
    #             [0,0,-0.15,0,0,np.pi/4]]
    # R,t = handEyeCalibrate(refPose, robotPose)
    print("estimation result: ", R,t)
    # np.save("transformation", [R,t])
    np.save("rotation", R)
    np.save("translation", t)