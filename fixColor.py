import numpy as np
import cv2

filename = input("input filename to fix: ")
color = cv2.imread("data\\"+filename+"_rgb.png")
b = color[:,:,0]
g = color[:,:,1]
r = color[:,:,2]
color = np.concatenate((r[:,:,np.newaxis],g[:,:,np.newaxis],b[:,:,np.newaxis]),axis=2)

cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
cv2.imshow("Align Example", color)
key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
if key & 0xFF == ord('q') or key == 27:
    cv2.destroyAllWindows()
cv2.imwrite("data\\"+filename+"_rgb.png",color)
