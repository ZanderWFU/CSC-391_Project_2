# CSC-391_Project_2
Local Feature Extraction, Detection and Matching

The Video_Edge.py file is for the "Edges and Corners in Video" section of the assignment, it can run either Canny Edge Detection or Harris
Corner Detection on input from the camera.
I implemented a control scheme to make it easier to control:

p - take a screenshot of the video
m - switch variable to adjust
. - increment variable
, - decrement variable
h - switch to adjusting maxVal (Canny Only)
l - switch to adjusting minVal (Canny Only)
o - toggle between outline and overlay display (Canny Only)

NOTE: Remaining part of this project requires "opencv-python 3.4.2.17" and opencv-contrib-python==3.4.2.17"
As newer versions have removed SIFT functionality

The Sift_Key.py file is for the "SIFT Descriptors and Scaling" section of the assignment, it will detect and display keypoints on image
with functionality to specify parameters

The Matching.py will match keypoints in two images from either harris corner detection or just a standard SIFT
