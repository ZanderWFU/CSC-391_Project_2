import numpy as np
import cv2
from matplotlib import pyplot as plt

#Settings:
CORNER_MODE = False #Use Keypoints from Corner Harris
INPUT_MODE = True #if true, there will be input prompts, otherwise it just goes with what's in this file

input_1 = 'wall_pic.jpeg'
input_2 = 'wall_pic_far.jpeg'



if INPUT_MODE:
    print("Input first image file name:")
    input_1 = input()
    print("Input second image file name:")
    input_2 = input()
    print("Use SIFT or Harris Corner Keypoints?:")
    mode_dict = {'sift': False, 's': False, 'harris': True, 'corner': True, 'c': True}
    invalid = True
    while (invalid):
        mode_input = input()
        mode_input = mode_input.lower()
        if mode_input in mode_dict:
            invalid = False
    CORNER_MODE = mode_dict[mode_input]

    print("Running Now...")

image_1 = cv2.imread(input_1) #Read images
image_2 = cv2.imread(input_2)

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY) #Convert to Grayscale
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

Sift = cv2.xfeatures2d.SIFT_create() #Create SIFT object

if CORNER_MODE:
    h_array = [2, 3, 0.04] #array for harris corners
    dst_1 = cv2.cornerHarris(gray_1, h_array[0], h_array[1], h_array[2])
    dst_2 = cv2.cornerHarris(gray_2, h_array[0], h_array[1], h_array[2])

#converts the output of cornerHarris into usable KeyPoints
    kp1 = np.argwhere(dst_1 > 0.04 * dst_1.max())
    kp2 = np.argwhere(dst_2 > 0.04 * dst_2.max())
    kp1 = [cv2.KeyPoint(x[1], x[0], 1) for x in kp1]
    kp2 = [cv2.KeyPoint(x[1], x[0], 1) for x in kp2]

else:
    kp1 = Sift.detect(gray_1, None)
    kp2 = Sift.detect(gray_2, None)



_, des1 = Sift.compute(gray_1, kp1)
_, des2 = Sift.compute(gray_2, kp2)

bf = cv2.BFMatcher() #Creates the matcher

matches = bf.match(des1,des2) #do the matching
matches = sorted(matches, key = lambda x:x.distance) #sort the matches



image_3 = cv2.drawMatches(image_1,kp1,image_2,kp2,matches[:10], flags=2, outImg=gray_1)

image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)

plt.imshow(image_3)
plt.xticks([]), plt.yticks([])
plt.show()
