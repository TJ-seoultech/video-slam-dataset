import cv2
import time
import argparse
import os
import re

parser = argparse.ArgumentParser(description='input one video for aruco detection(DSO SLAM), output images.zip and times.txt(timestamp) for gamma calibraion(dso slam) or for orb slam. \n output default directory: ./dataset/')
parser.add_argument('-file', required=True, help = 'video file path')
parser.add_argument('-slam', required=False, default= 'dso', help = 'choose output option, dso or orb')
parser.add_argument('-dir', required=False, default = './data/', help = 'output directory')

args=parser.parse_args()

print("{:-^60}".format("<< input arguments >>"))
print("\n")
print("{:<15} : {:<25}".format('-file', args.file))
print("{:<15} : {:<25}".format('-slam', args.slam))
print("{:<15} : {:<25}".format('-dir', args.dir))

filepath = args.file
slam = args.slam
dir = args.dir

def extract(filepath):

    cap = cv2.VideoCapture(filepath)

    if not cap.isOpened():
        print("Could not Open :", filepath)
        exit(0)
    elif '.' not in filepath:
        print("Missing file format. The file argument should be sample_video.mp4")

    #불러온 비디오 파일의 정보 출력
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("length(비디오 파일의 총 프레임 수) :", length)
    print("width(프레임 가로 크기) :", width)
    print("height(프레임 세로 크기) :", height)
    print("fps(초당 프레임 수) :", fps)

    if slam == 'dso':
        
        f = open("times.txt", 'w')
        stamp = time.time()
        exposure_time = round(1/(2*fps)*1000,1)
        # foldername = filepath.split('.')[1]
        # print(filepath.split('.'))
        # print('data/{}'.format(foldername)) # data/\LabSlam
        datasetName = re.sub(r"[^a-zA-Z0-9]", "", filepath.split('.')[-2])
        os.mkdir('data/{}'.format(datasetName))
        os.mkdir('data/{}/original_images'.format(datasetName))
    

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True: 
                # 이 조건문을 추가하지 않으면 오류가 난다. error: (-215)size.width>0 && size.height>0 in function imshow
                # 보통은 파일 경로를 잘못 지정하고 imread를 할 때 나는 오류이지만, VideoCapture 함수를 사용하는 경우에는, ret(return value 검사)를 하지 않아 발생한다.
                
                cv2.imshow('Press q to exit.', frame)
                num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.imwrite('data/{}/original_images/{:0>5}.png'.format(datasetName,num), frame)
                print('Saved frame file name : {}.png'.format(num))
                data = "{:0>5} {:0<21} {}\n".format(num, stamp, exposure_time)
                f.write(data)
                stamp += exposure_time/1000

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        f.close()
        cap.release()
        cv2.destroyAllWindows()


    elif slam == 'orb':
        print("not prepared yet...!")

