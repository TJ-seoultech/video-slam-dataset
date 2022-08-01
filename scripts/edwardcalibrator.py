"""
Aruco camera calibration
"""

# Import required packages:
from dis import dis
from math import dist
import time
import cv2
import numpy as np
import pickle

def arucocalib(filepath, datasetName, dir, slam):
    # Create dictionary and board object:
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
    board = cv2.aruco.CharucoBoard_create(7, 5, .0029, .00235, dictionary)

    # Create board image to be used in the calibration process:
    image_board = board.draw((781, 200 * 3))

    # Write calibration board image:
    cv2.imwrite('charuco.png', image_board)

    # Create VideoCapture object:
    cap = cv2.VideoCapture(filepath)

    all_corners = []
    all_ids = []
    counter = 0

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(length):

        # Read frame from the webcam:
        ret, frame = cap.read()

        # Convert to grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers:
        res = cv2.aruco.detectMarkers(gray, dictionary)

        if len(res[0]) > 0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
                all_corners.append(res2[1])
                all_ids.append(res2[2])

            cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1

    # Calibration can fail for many reasons:
    try:
        cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
    except:
        cap.release()
        print("Calibrationcould not be done ...")

    # Get the calibration result:
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal
    print("camera matrix")
    print(cameraMatrix)
    print("distCoeffs")
    print(distCoeffs)

    # Save the camera parameters:
    # f = open('calibration2.pckl', 'wb')
    # pickle.dump((cameraMatrix, distCoeffs), f)
    # f.close()
    f = open('{}/{}/camera.txt'.format(dir, datasetName), 'w')
    f.write('{} {} {} {} {} {} {} {}\n'.format(
        round(cameraMatrix[0][0]/1280,4),
        round(cameraMatrix[0][2]/720,4),
        round(cameraMatrix[1][1]/1280,4),
        round(cameraMatrix[1][2]/1280,4),
        round(distCoeffs[0][0]),
        round(distCoeffs[0][1]),
        round(distCoeffs[0][4]),
        0
    ))
    f.write("1280 720\n")
    f.write("crop\n")
    f.write("1280 720")
    f.close()

    # Release everything:
    cap.release()
    cv2.destroyAllWindows()