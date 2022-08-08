# -*- coding: utf-8 -*-
import cv2
import time
import argparse
import os
import re
import zipfile



def check_Data_hierarchy(dir):
    dir = dir.replace("\\","").replace(".","")
    if dir in os.listdir("./"):
        print("directory is ok")
    else:
        print("Directory '{}' doesn't exist in project root folder.\n Creating database folder named '{}/ ' on the project root directory...\n".format(dir, dir))
        os.mkdir("{}".format(dir))
    time.sleep(5)



def extract(filepath,slam,dir):
    
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

    print("{:<9}{:<18} : {}".format("length","(비디오 파일의 총 프레임 수)", length))
    print("{:<9}{:<18} : {}".format("width","(프레임 가로 크기)", width))
    print("{:<9}{:<18} : {}".format("height","(프레임 세로 크기)", height))
    print("{:<9}{:<18} : {}".format("fps","(초당 프레임 수)", fps))
    time.sleep(4.3)

    if slam == 'dso':
        
        # foldername = filepath.split('.')[1]
        # print(filepath.split('.'))
        # print('{}/{}'.format(foldername)) # {}/ \LabSlam
        datasetName = re.sub(r"[^a-zA-Z0-9]", "", filepath.split('.')[-2])
        os.mkdir('{}/{}'.format(dir,datasetName))
        os.mkdir('{}/{}/original_images'.format(dir,datasetName))
        f = open("{}/{}/times.txt".format(dir,datasetName), 'w')
        stamp = time.time()
        exposure_time = round(1/(2*fps)*1000,1)


        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True: 
                # 이 조건문을 추가하지 않으면 오류가 난다. error: (-215)size.width>0 && size.height>0 in function imshow
                # 보통은 파일 경로를 잘못 지정하고 imread를 할 때 나는 오류이지만, VideoCapture 함수를 사용하는 경우에는, ret(return value 검사)를 하지 않아 발생한다.
                
                cv2.imshow('Press q to exit.', frame)
                num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.imwrite('{}/{}/original_images/{:0>5}.png'.format(dir,datasetName,num), frame)
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
    


def resize(datasetName, dir, width=1280, height=720):
    path = '{}/{}'.format(dir,datasetName)
    images = os.listdir(path+'/original_images')
    os.mkdir(path+'/reduced_images')

    for image in images:
        src = cv2.imread('{}/original_images/{}'.format(path,image), cv2.IMREAD_COLOR)
        print("resizing {}...".format(os.path.join(path,'original_images',image)))
        dst = cv2.resize(src, dsize=(640,360), interpolation=cv2.INTER_AREA)
        cv2.imwrite('{}/reduced_images/{}'.format(path,image), dst)



def zip(datasetName, dir):
    os.chdir("{}/{}".format(dir,datasetName))
    zip_file = zipfile.ZipFile("360p-images.zip", "w")
    cwd = os.chdir("reduced_images")
    print("compressing resized images...360p images.")
    for file in os.listdir("./"):
        if file.endswith('.png'):
            zip_file.write(os.path.join(file), compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()

    cwd = os.chdir("../")
    print("compressing original images...720p images.")
    zip_file = zipfile.ZipFile("720p-images.zip", "w")
    cwd = os.chdir("original_images")

    for file in os.listdir("./"):
        if file.endswith('.png'):
            zip_file.write(os.path.join(file), compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()


