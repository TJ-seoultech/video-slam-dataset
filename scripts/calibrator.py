import numpy as np
import cv2, PIL, os
from cv2 import aruco
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import re

# import numpy
# import cv2
# from cv2 import aruco
# import pickle 
import glob
import os


# failure
# board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)

def fish_eye_calib(datasetName, slam, dir):

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)

    datadir = "{}/{}/reduced_images/".format(dir, datasetName)
    images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ])
    im = PIL.Image.open(images[0])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)


    def read_chessboards(images):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in images:
            print("=> Processing image {0}".format(im))
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
            
            if len(corners)>0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(gray, corner, 
                                    winSize = (3,3), 
                                    zeroZone = (-1,-1), 
                                    criteria = criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)        
                if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])              
            
            decimator+=1   

        imsize = gray.shape
        return allCorners,allIds,imsize


    def calibrate_camera(allCorners,allIds,imsize):   
        """
        Calibrates the camera using the dected corners.
        """
        print("CAMERA CALIBRATION")
        
        cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                    [    0., 1000., imsize[1]/2.],
                                    [    0.,    0.,           1.]])

        distCoeffsInit = np.zeros((5,1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO) 
        #flags = (cv2.CALIB_RATIONAL_MODEL) 
        (ret, camera_matrix, distortion_coefficients0, 
        rotation_vectors, translation_vectors,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics, 
        perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=allCorners,
                        charucoIds=allIds,
                        board=board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors



    allCorners,allIds,imsize=read_chessboards(images)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)

    print("{:^35}".format("camera matrix"))
    print(mtx)
    print("{:^35}".format("distortion"))
    print(dist)

# works, but make things worse
# board = aruco.CharucoBoard_create()
def charuco_calib(filepath, dir, captures):
    '''
    calibrate chAruco board video
    '''
    # ChAruco board variables
    CHARUCOBOARD_ROWCOUNT = 7
    CHARUCOBOARD_COLCOUNT = 5 
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_50)

    # Create constants to be passed into OpenCV and Aruco methods
    CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)

    # Corners discovered in all images processed
    corners_all = []

    # Aruco ids corresponding to corners discovered 
    ids_all = [] 

    # Determined at runtime
    image_size = None 

    # This requires a video taken with the camera you want to calibrate
    cap = cv2.VideoCapture(filepath)

    # The more valid captures, the better the calibration
    validCaptures = 0

    # Loop through frames
    while cap.isOpened():

        # Get frame
        ret, img = cap.read()

        # If camera error, break
        if ret is False:
            break

        # Grayscale the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find aruco markers in the query image
        corners, ids, _ = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)
        
        # If none found, take another capture
        if ids is None:
            continue

        # Outline the aruco markers found in our query image
        img = aruco.drawDetectedMarkers(
            image=img, 
            corners=corners)

        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

        # If a Charuco board was found, collect image/corner points
        # Requires at least 20 squares for a valid calibration image
        if response > 20:
            # Add these corners and ids to our calibration arrays
            corners_all.append(charuco_corners)
            ids_all.append(charuco_ids)
            
            # Draw the Charuco board we've detected to show our calibrator the board was properly detected
            img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
        
            # If our image size is unknown, set it now
            if not image_size:
                image_size = gray.shape[::-1]
            
            # Reproportion the image, maxing width or height at 1000
            proportion = max(img.shape) / 1000.0
            img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

            # Pause to display each image, waiting for key press
            cv2.imshow('Charuco board', img)
            if cv2.waitKey(0) == ord('q'):
                break0= 1
            if validCaptures == captures:
                break

    # Destroy any open CV windows
    cv2.destroyAllWindows()

    # Show number of valid captures
    print("{} valid captures".format(validCaptures))

    if validCaptures < captures:
        print("Calibration was unsuccessful. We couldn't detect enough charucoboards in the video.")
        print("Perform a better capture or reduce the minimum number of valid captures required.")
        exit()

    # Make sure we were able to calibrate on at least one charucoboard
    if len(corners_all) == 0:
        print("Calibration was unsuccessful. We couldn't detect charucoboards in the video.")
        print("Make sure that the calibration pattern is the same as the one we are looking for (ARUCO_DICT).")
        exit()
    print("Generating calibration...")

    # Now that we've seen all of our images, perform the camera calibration
    calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
            
    # Print matrix and distortion coefficient to the console
    print("Camera intrinsic parameters matrix:\n{}".format(cameraMatrix))
    print("\nCamera distortion coefficients:\n{}".format(distCoeffs))
            
    datasetName = re.sub(r"[^a-zA-Z0-9]", "", filepath.split('.')[-2])

    f = open("{}/{}/EquiDistant_camera.txt".format(dir,datasetName), 'w')
    f.write('{} {} {} {} {} {} {} {} {}\n'.format(
        'EquidDstant',
        round(cameraMatrix[0][0]/1280,5),
        round(cameraMatrix[1][1]/720,5),
        round((cameraMatrix[0][2]+0.5)/1280,5),
        round((cameraMatrix[1][2]+0.5)/720,5),
        round(distCoeffs[0][0],5),
        round(distCoeffs[0][1],5),
        round(distCoeffs[0][4],5),
        0)
        )
    f.write("1280 720\n")
    f.write("crop\n")
    f.write("1280 720")
    f.close()


    f = open("{}/{}/RadTan_camera.txt".format(dir,datasetName), 'w')
    f.write('{} {} {} {} {} {} {} {} {}\n'.format(
        'RadTan',
        round(cameraMatrix[0][0]/1280,5),
        round(cameraMatrix[1][1]/720,5),
        round((cameraMatrix[0][2]+0.5)/1280,5),
        round((cameraMatrix[1][2]+0.5)/720,5),
        round(distCoeffs[0][0],5),
        round(distCoeffs[0][1],5),
        round(distCoeffs[0][2],5),
        round(distCoeffs[0][2],5))
        )
    f.write("1280 720\n")
    f.write("crop\n")
    f.write("1280 720")
    f.close()


# board = cv2.aruco.CharucoBoard_create(3,3,.025,.0125,dictionary)
def charuco_calib_2(filepath, dir, captures):

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard_create(7,5,.004,.002,dictionary)
    img = board.draw((200*3,200*3))

    #Dump the calibration board to a file
    cv2.imwrite('charuco.png',img)


    #Start capturing images for calibration
    cap = cv2.VideoCapture(filepath)

    allCorners = []
    allIds = []
    decimator = 0
    for i in range(300):

        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray,dictionary)

        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%3==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

            cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        decimator+=1

    imsize = gray.shape

    #Calibration fails for lots of reasons. Release the video if we do
    try:
        cal = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
        print("00000000")
        print(cal)
    except:
        cap.release()
        print("1")
    cap.release()
    cv2.destroyAllWindows()


def charuco_calib_3(filepath, dir, captures):
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
    images = glob.glob('calibration_iphone11/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        # image resize
        img = cv2.resize(img, (504, 378))
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