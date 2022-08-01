import os
import cv2


path = 'cat'
images = os.listdir(path)

for image in images:
    src = cv2.imread('{}/{}'.format(path,image), cv2.IMREAD_COLOR)
    dst = cv2.resize(src, dsize=(640,360), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./cat/images/{}'.format(image), dst)


'''
img = cv2.imread("cat.jpg")
cv2.imshow("Press q to exit.", img)
cv2.waitKey(delay=None)
cv2.imwrite("{}/{}".format("cat.jpg".split('.')[0],"cat2.jpg"), img)

file_path="cat"
zip_file = zipfile.ZipFile(file_path + "\\output3.zip", "w")

file_path = os.chdir(file_path)
for file in os.listdir("./"):
    if file.endswith('.jpg'):
        zip_file.write(os.path.join(file), compress_type=zipfile.ZIP_DEFLATED)

zip_file.close()
'''

# src = cv2.imread("Image/champagne.jpg", cv2.IMREAD_COLOR)

# dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)

