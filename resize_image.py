import cv2
from os import listdir
import os


path ="dataset/progress""

for forder_name in listdir(path):
    os.mkdir(forder_name)
    for file_name in listdir(path + "/" + forder_name):
        im = cv2.imread(path + "/" + forder_name + "/" + file_name)
        im = cv2.resize(im, (256,256), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(forder_name, file_name),im )
        