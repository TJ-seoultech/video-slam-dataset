import cv2

img = cv2.imread("image/cat.jpg")
cv2.imshow("img", img)
print("img.shape = {0}".format(img.shape))
cv2.waitKey()