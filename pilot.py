from importlib.metadata import files
import os
import argparse
import time
import re
import cv2
import numpy as np
import glob


def check_existence(input,type):
    if input == None:
        print(f"{type} input None.")
        return False
    elif os.path.exists(input):
        print(f"{type} input '{input}' found")
        return True
    else:
        print(f"{type} input not found")
        return False
    
    
def check_runnable(c,d):
    if (c or d) and not(c and d):
        return True
    else:
        return False
        
        

def choose_video(a,b):
    if a and b:
        print("dataset with file and folder")
        return 3
    elif a and not b:
        print("dataset with file")
        return 1
    elif not a and b:
        print("dataset with folder")
        return 2
    else:
        print("only calibration")
        return 0


# check device folder exists and mkdir it
def check_device_folder(device):
    if not os.path.exists(device):
        os.mkdir(device)    # Create device folder
        
    if not os.path.exists(os.path.join(device,'videos')):
        os.mkdir("{}/{}".format(device,'videos'))
    else: 
        print("device folder check ok. {}/{}".format(device,'videos')) 
    
    # check_existence device calibration folder
    
    if not os.path.exists(os.path.join(device,'calibration')):
        os.mkdir("{}/{}".format(device,'calibration'))
    else:
        print("device folder check ok. {}/{}".format(device,'calibration')) # Check existence of calibration folder


def read_video(dir, video):
    try:
        cap = cv2.VideoCapture(video)
    except:
        print("can't read video")
    else:
        print("read video")
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("{:<9}{:<18} : {}".format("length","(비디오 파일의 총 프레임 수)", length))
        print("{:<9}{:<18} : {}".format("width","(프레임 가로 크기)", width))
        print("{:<9}{:<18} : {}".format("height","(프레임 세로 크기)", height))
        print("{:<9}{:<18} : {}".format("fps","(초당 프레임 수)", fps))
        time.sleep(2.3)
        
        # datasetName = re.sub(r"[^a-zA-Z0-9]", "", video.split('.')[-2])
        # erarse backslash from file name
        datasetName = file.replace('\\', '').split('.')[-2]
        os.mkdir('{}/videos/{}'.format(dir,datasetName))
        os.mkdir('{}/videos/{}/images'.format(dir,datasetName))
        f = open("{}/videos/{}/times.txt".format(dir,datasetName), 'w') # create timestamp for dso slam
        g = open("{}/videos/{}/orbtimes.txt".format(dir,datasetName),'w') # Create timestamp for orb slam
        stamp = time.time()
        exposure_time = round(1/(2*fps)*1000,1)

        print("{}/videos/{} capturing...".format(dir,datasetName))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True: 
                # 이 조건문을 추가하지 않으면 오류가 난다. error: (-215)size.width>0 && size.height>0 in function imshow
                # 보통은 파일 경로를 잘못 지정하고 imread를 할 때 나는 오류이지만, VideoCapture 함수를 사용하는 경우에는, ret(return value 검사)를 하지 않아 발생한다.
                
                # cv2.imshow('Press q to exit.', frame)
                num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                # cv2.imwrite('{}/videos/{}/images/{:0>5}.png'.format(dir,datasetName,num), frame)
                cv2.imwrite('{}/videos/{}/images/{:0<19}.png'.format(dir,datasetName,stamp), frame)

                print('ret : {} . Saved frame number : {}'.format(ret, num))
                dso_data = "{:0>5} {:0<21} {}\n".format(num, stamp, exposure_time)
                orb_data = "{:0<19}\n".format(stamp)
                f.write(dso_data)
                g.write(orb_data)
                stamp += exposure_time/1000

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                    
            if num == fps:
                break
        f.close()
        g.close()
        cap.release()
        cv2.destroyAllWindows()


def choose_calibration(c,d):
    if c:
        return 0
    else:
        print("Use one")
        return 1


def use_calibration(calibration):
    # use preprocessed calibration
    os.path.exists(calibration)
    
    
def calibrate(dir, images):
    # calibration images from argument images

    # 체커보드의 차원 정의
    CHECKERBOARD = (6,9) # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
    objpoints = []
    # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
    imgpoints = []

    # 3D 점의 세계 좌표 정의
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
    images = glob.glob(f'{images}/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        # image resize
        img = cv2.resize(img, (1280, 720))
        # print(img.shape)
        # 그레이 스케일로 변환
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 체커보드 코너 찾기
        # 이미지에서 원하는 개수의 코너가 발견되면 ret = true
        ret, corners = cv2.findChessboardCorners(gray,
                                                CHECKERBOARD,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # 원하는 개수의 코너가 감지되면,
        # 픽셀 좌표 미세조정 -> 체커보드 이미지 표시
        if ret == True:
            objpoints.append(objp)
            # 주어진 2D 점에 대한 픽셀 좌표 미세조정
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            # 코너 그리기 및 표시
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    h,w = img.shape[:2] # 480, 640

    # 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n") # 내부 카메라 행렬
    print(mtx)

    print("dist : \n") # 렌즈 왜곡 계수(Lens distortion coefficients)
    print(dist)

    print("rvecs : \n") # 회전 벡터
    print(rvecs)

    print("tvecs : \n") # 이동 벡터
    print(tvecs)
    
    c = open("{}/calibration/EquiDistant_camera.txt".format(dir), 'w')
    c.write('{} {} {} {} {} {} {} {} {}\n'.format(
        'EquidDstant',
        round(mtx[0][0]/1280,5),
        round(mtx[1][1]/720,5),
        round((mtx[0][2]+0.5)/1280,5),
        round((mtx[1][2]+0.5)/720,5),
        round(dist[0][0],5),
        round(dist[0][1],5),
        round(dist[0][4],5),
        0)
        )
    c.write("1280 720\n")
    c.write("crop\n")
    c.write("1280 720")
    c.close()


    c = open("{}/calibration/camera.txt".format(dir), 'w')
    c.write('{} {} {} {} {} {} {} {} {}\n'.format(
        'RadTan',
        round(mtx[0][0]/1280,5),
        round(mtx[1][1]/720,5),
        round((mtx[0][2]+0.5)/1280,5),
        round((mtx[1][2]+0.5)/720,5),
        round(dist[0][0],5),
        round(dist[0][1],5),
        round(dist[0][2],5),
        round(dist[0][3],5))
        )
    c.write("1280 720\n")
    c.write("crop\n")
    c.write("1280 720")
    c.close()

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
    description='input one video file, output images.zip and times.txt(timestamp) for gamma calibraion(dso slam) or for orb slam. One required argument, two are optional.',
    epilog='Input aruco detection video for DSO SLAM.')
    parser.add_argument('-video', required=False, default = None, help = '(Optional) video file path')
    parser.add_argument('-videofolder', required=False, default = None, help = '(Optional) video folder path')
    parser.add_argument('-images', required=False, default = None, help = '(Optional) images(for calibration) path')
    parser.add_argument('-calibrator', required=False, default= 'nocalib', help = '(Optional) choose camera calibration file')
    parser.add_argument('-slam', required=False, default= 'dso', help = '(Optional) choose output option, dso or orb')
    parser.add_argument('-device', required=False, default = 'data', help = '(Optional) output directory, default is data')

    args=parser.parse_args()

    video = args.video
    videofolder = args.videofolder
    images = args.images
    calibration = args.calibrator
    slam = args.slam
    device = args.device.replace('/','')

    check_device_folder(device)

    
    videoBool = check_existence(video, "video")
    videofolderBool = check_existence(videofolder, "videofolder")
    imagesBool = check_existence(images, "images")
    calibrationBool = check_existence(os.path.join(device,'calibration',calibration), "calibration")
    slamBool = check_existence(slam, "slam")
    deviceBool = check_existence(device, "device")

    print("\n")
    print("{:<25} : {}".format("videoBool",videoBool))
    print("{:<25} : {}".format("videofolderBool",videofolderBool))
    print("{:<25} : {}".format("imagesBool",imagesBool))
    print("{:<25} : {}".format("calibrationBool",calibrationBool))
    print("{:<25} : {}".format("slamBool",slamBool))
    print("{:<25} : {}".format("deviceBool",deviceBool))

    perform = check_runnable(imagesBool, calibrationBool)

    if perform == True:
        print("\n")
        # calibrator = choose_calibration(imagesBool, calibrationBool)
        calibmode = choose_calibration(imagesBool, calibrationBool)
        datamode = choose_video(videoBool, videofolderBool)
        videos = list()
        
        if datamode == 1:
            # only one video is available
            videos.append(re.sub(r"[^a-zA-Z0-9]", "", video.split('.')[-2]))
            print(videos)
            check_device_folder(device)
            print("device is available")
            if calibmode == True:
                use_calibration(calibration)
            else:
                print("calibration")
                calibrate(device, images)
            print("read and extract video")
            read_video(device, video)
            
            
            
        elif datamode == 2:
            files=list()
            # read videos from video folder and capture videos
            for file in os.listdir(videofolder):
                if file.endswith(".mp4"):
                    files.append(file)
            
            print(files)
            if not os.path.exists(os.path.join(device,'videos',videofolder)):
                os.mkdir(os.path.join(device,'videos',videofolder))
            for file in files:
                file = os.path.join(videofolder, file)
                print(file)
                read_video(device, file)
                
            if calibmode == True:
                use_calibration(calibration)
            else:
                print("calibration")
                calibrate(device, images)
        
        elif datamode == 3:
            print("둘중 하나만 하시오.")
            
        else:
            calibrate(device, images)
                

            # if calibrationBool == True:
            #     os.path.exists(calibration)         
    else:
        print("This logic can't be executable. Arguments are set wrong. check inputs.")
    