import numpy as np
import cv2
import glob
import pyrealsense2 as rs
import matplotlib.pyplot as plt


def capture():
    # Declare RealSense pipeline, encapsulating the actual device and sensors
    align = rs.align(rs.stream.color)
    pipe = rs.pipeline()
    config = rs.config()
    # Enable depth stream
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)  #, 640, 480, rs.format.bgr8, 30
    count = 0
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

            images = color
            cv2.imshow('img', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            if key == ord(' '):
                count += 1
                cv2.imwrite("chessboard_"+str(count)+".png", images)

        
    finally:
        pipe.stop()

def calibrate():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) * 30
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('calibrate/*.png')
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("found")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    return objpoints, imgpoints, gray

capture()
# a,ret = cv2.Rodrigues(np.array([0,np.pi,0]))
# print(a)
# objpoints, imgpoints, gray = calibrate()
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(ret, mtx, dist, rvecs, tvecs)