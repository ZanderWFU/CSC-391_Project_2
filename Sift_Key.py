import cv2
from matplotlib import pyplot as plt

#Settings
SHOW_PLT = True #show with plt or just cv2 (better for big images)
COLOR_DISPLAY = True #Display keypoints over color image or grayscale one, (easier to see colorful keypoints)
SAVE_IMAGE = True #save the image?
INPUT_MODE = True #if true, there will be input prompts, otherwise it just goes with what's in this file

input_file = 'wall_pic_right.jpeg'

key_limit = 0 #nfeatures
edge = 10 #edgeThreshold
contrast = 0.03 #contrastThreshold

if INPUT_MODE:
    print("Input image file name:")
    input_file = input()
    print("Input nFeaturesL (0 for no limit)")
    key_limit = int(input())
    print("Input edgeThreshold: (default = 10)")
    edge = int(input())
    print("Input contrastEdge: (default = 0.03)")
    contrast = float(input())
print("Running...")

image = cv2.imread(input_file) #Reads the image
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converts it to Gray

sift = cv2.xfeatures2d.SIFT_create(nfeatures = key_limit, contrastThreshold = contrast, edgeThreshold = edge) #Creates the SIFTer

kp = sift.detect(image_gray,None) #gets the keypoints from the image
print(f'Keypoints: {len(kp)}')

if not COLOR_DISPLAY:
    image = image_gray #converts
image = cv2.drawKeypoints(image, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #draws the keypoints onto the image

if SAVE_IMAGE:
    cv2.imwrite(f'SIFT_output_{contrast}.jpg', image)

if SHOW_PLT:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.show()
else:
    cv2.imshow("Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()