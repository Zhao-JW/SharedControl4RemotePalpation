# First import the library
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import cv2

filename = "data\\" + input("input file name: ")

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