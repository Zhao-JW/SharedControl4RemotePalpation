# SharedControl4RemotePalpation
Personal fourth year project repository: Shared Control for Underactuated Remote Plapation - A Comparative Study

The codes, videos, images and other documents are kept in a GitHub repository for backup and version control (https://github.com/Zhao-JW/SharedControl4RemotePalpation.git).

* Data: the data folder contains two sub-folders, data and csvdata. The data sub-folder has the  RGB-D image taken throughout the project. The csvdata contains the original data collected during the user experiment.
* STL: the STL folder contains the stl file of the past versions of the camera mount. This is used to mount the tactile sensor and the depth camera onto the UR5 robot.
*  Mic presentation \& Lent presentation: original PowerPoint and a few important videos and images used in the presentation.
*  Video: videos made to demonstrate the experiment process, for use in the final presentation and thesis. These are not experiment recordings. 
*  SHAREDCONTROL: main code for the shared control algorithm, this will be imported as a class for use in the remote palpation program.
*  angle: code used to reconstruct the tilt angle of a flat phantom,  used to verify the reliability of the visual reconstruction.
*  capture: code used to capture RGB-D image for testing and code prototyping at the initial stage of the project. 
*  cameraCalib/QR4robot: camera calibration code using chessboard calibration pattern/QR code, not used in the final system.
*  manualCalib: camera calibration code using four red calibration markers, this is the same as the algorithm used in the final system.
*  plotting code: a set of codes used to process the raw data and plot figures used in the final report, it includes: csv\_cluster, csv\_euclid, csv\_keys, csv\_main, csv\_process, csv\_tactile, tumor\_analysis
 
