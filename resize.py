import os
import cv2


path = 'C:/Users/ict-526-tj/Documents/mono_tum_dataset/Largeimages/'
images = os.listdir(path)

for image in images:
    src = cv2.imread(path+image, cv2.IMREAD_COLOR)
    dst = cv2.resize(src, dsize=(640,360), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./images/{}'.format(image), dst)



# src = cv2.imread("Image/champagne.jpg", cv2.IMREAD_COLOR)

# dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

