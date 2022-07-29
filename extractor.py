import cv2
import time
import datetime

filepath = 'LabSlam.mp4'
cap = cv2.VideoCapture(filepath)

if not cap.isOpened():
    print("Could not Open :", filepath)
    exit(0)

#불러온 비디오 파일의 정보 출력
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("length(비디오 파일의 총 프레임 수) :", length)
print("width(프레임 가로 크기) :", width)
print("height(프레임 세로 크기) :", height)
print("fps(초당 프레임 수) :", fps)

f = open("times.txt", 'w')
iterateN = 1 # DSO-SLAM 용으로 사진 이름을 1부터 순서대로 넣는다.

stamp = time.time()
exposure_time = round(1/(2*fps)*1000,1)

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True: # 이 조건문을 추가하지 않으면 오류가 난다. error: (-215)size.width>0 && size.height>0 in function imshow
        # 보통은 파일 경로를 잘못 지정하고 imread를 할 때 나는 오류이지만, VideoCapture 함수를 사용하는 경우에는, ret(return value 검사)를 하지 않아 발생한다.
        cv2.imshow('video', frame)



        num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if (num):  # 앞서 불러온 fps 값을 사용하여 1초마다 추출
            # cv2.imwrite('images/{0:0>5}.png'.format(int(num)), frame)
            print('Saved frame file name : {}.png'.format(num))
            data = "{:0>5} {:0<21} {}\n".format(num, stamp, exposure_time)
            f.write(data)
            stamp += exposure_time/1000

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
f.close()
cap.release()
cv2.destroyAllWindows()