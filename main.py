import argparse
from ast import arg
from os import lseek
import os
import cv2
import argparse
import re
import scripts.filemanager as manager
import scripts.edwardcalibrator as calib


parser = argparse.ArgumentParser(
    description='input one video file, output images.zip and times.txt(timestamp) for gamma calibraion(dso slam) or for orb slam. One required argument, two are optional.',
    epilog='Input aruco detection video for DSO SLAM.')
parser.add_argument('-file', required=True, help = 'video file path(Required)')
parser.add_argument('-slam', required=False, default= 'dso', help = 'choose output option, dso or orb(Optional)')
parser.add_argument('-dir', required=False, default = './data/', help = 'output directory, default is data(Optional)')

args=parser.parse_args()



print("{:-^60}".format("<< input arguments >>"))
print("\n")
print("{:<15} : {:<25}".format('-file', args.file))
print("{:<15} : {:<25}".format('-slam', args.slam))
print("{:<15} : {:<25}".format('-dir', args.dir))

filepath = args.file        # You can assign variable manually.
slam = args.slam            # You can assign variable manually.
dir = args.dir              # You can assign variable manually.


# filepath = "20220731_194514.mp4"
# slam = 'dso'
# dir = 'galaxyA52s'


datasetName = re.sub(r"[^a-zA-Z0-9]", "", filepath.split('.')[-2])
root = os.getcwd()
manager.check_Data_hierarchy(dir)
manager.extract(filepath, slam, dir)
manager.resize(datasetName, dir, width=640, height=360)
manager.zip(datasetName, dir)
calib.arucocalib(filepath, datasetName, dir, slam)