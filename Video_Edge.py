import cv2

#Settings:
PROMPT_MODE = False #when I'm testing repeatedly, it's quicker to change booleans here
RUN_ON_VIDEO = True #if False, runs on image
corner = True #true for Harris false for canny


image_name = "wall_pic_far.jpeg"
if not RUN_ON_VIDEO:
    in_image = cv2.imread(image_name)
if PROMPT_MODE:
    print("Run Canny Edge Detection or Harris Corner Detection?")
    mode_dict = {'canny': False, 'edge': False, 'e': False, 'harris': True, 'corner': True, 'c': True}
    invalid = True
    while (invalid):
        mode_input = input()
        mode_input = mode_input.lower()
        if mode_input in mode_dict:
            invalid = False
    corner = mode_dict[mode_input]

if corner:
    print("Running Harris Corner Detection")
else:
    print("Running Canny Edge Detection")

if RUN_ON_VIDEO:
    cap = cv2.VideoCapture(0)

low = 100
high = 140
pic_index = 0
low_mode = False # True: minVal, False: maxVal
if corner:
    harris_mode = 0 #changes what is to be adjusted, 0: blocksize, 1: ksize, 2:k

    h_array = [2, 3, 0.04]
    inc_dict = {0: 1, 1: 2, 2: 0.001}
    n_dict = {0: "Block Size", 1: "k Size", 2: "k"}

overlay = False #overlays the edges onto the image


if RUN_ON_VIDEO:
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
if not corner:
    print(f'minVal: {low} maxVal: {high}')
else:
    print(f"Block Size: {h_array[0]} k Size: {h_array[1]} k: {h_array[2]}")
    print(f"Adjusting {n_dict[harris_mode]}")

while True:
    if RUN_ON_VIDEO:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    else:
        frame = in_image.copy()


    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if corner:
        cframe = cv2.cornerHarris(gframe, h_array[0], h_array[1], h_array[2])
        cframe = cv2.dilate(cframe, None)
    else:
        cframe = cv2.Canny(gframe, low, high, L2gradient=True) #runs canny edge detection

    if corner:
        frame[cframe > 0.04 * cframe.max()] = [0, 0, 255] #thresholding
    elif overlay:
        cframe = cv2.cvtColor(cframe, cv2.COLOR_GRAY2BGR)
        cframe[:, :, 0] = 0
        cframe[:, :, 2] = 0
        frame = cv2.addWeighted(frame, 0.7, cframe, 0.3, 0)

    else:
        frame = cframe

    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break
    if corner:
        if c==ord('p'): #p save the frame as an image
            filename = f'pic_{pic_index}.jpg'
            cv2.imwrite(filename, frame)
            pic_index += 1
            print(f'screenshot: {filename} taken')
        if c==ord('m'):
            harris_mode = harris_mode+1
            harris_mode = harris_mode % 3
            print(f"Adjusting {n_dict[harris_mode]}")
        if c==ord('.'):
            h_array[harris_mode] += inc_dict[harris_mode]
            print(f"Block Size: {h_array[0]} k Size: {h_array[1]} k: {h_array[2]}")
        if c==ord(','):
            h_array[harris_mode] += -inc_dict[harris_mode]
            if h_array[harris_mode] < 1 and harris_mode != 2:
                h_array[harris_mode] = 1
            print(f"Block Size: {h_array[0]} k Size: {h_array[1]} k: {h_array[2]}")
    else:
        if c == ord('h'): #h adjust max value
            low_mode = False
            print('Adjusting maxVal')
        if c == ord('l'): #l adjust min value
            low_mode = True
            print('Adjusting minVal')
        if c == ord('m'): #m toggle min/max mode
            if low_mode:
                print('Adjusting maxVal')
            else:
                print('Adjusting minVal')
            low_mode = not low_mode
        if c == ord('.'): #. increment by 10
            if low_mode:
                low += 10
            else:
                high += 10
            print(f'minVal: {low} maxVal: {high}')
        if c == ord(','): #, decrement by 10
            if low_mode:
                low += -10
            else:
                high += -10
            print(f'minVal: {low} maxVal: {high}')
        if c== ord('o'): #o toggle overlay mode
            if overlay:
                print('Outline Mode')
            else:
                print('Overlay Mode')
            overlay = not overlay
        if c==ord('p'): #p save the frame as an image
            filename = f'pic_min{low}_max{high}_{pic_index}.jpg'
            cv2.imwrite(filename, frame)
            pic_index += 1
            print(f'screenshot: {filename} taken')

if RUN_ON_VIDEO:
    cap.release()
cv2.destroyAllWindows()