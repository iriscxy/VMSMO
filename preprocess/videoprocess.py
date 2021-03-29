import cv2
import numpy as np
import os
import shutil

def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)

dirs = os.listdir('your video dir')
num = 0
suc = 0
for file in dirs:
    id = file.split('.')[0]
    time = id[0:6]
    path = 'the video images path you want to save images'
    dirs = os.listdir(path)
    num += 1
    if num % 1000 == 0:
        print('num: {}, suc: {}'.format(num, suc))
    if len(dirs) != 0:
        continue
    # os.mkdir(path)

    # 读取视频文件
    videoCapture = cv2.VideoCapture('your video file')

    # 读帧
    success, frame = videoCapture.read()
    i = 0
    timeF = 10
    j = 0
    max = 0
    tmp = 1000000
    while success:
        if j >= 200:
            break
        if (i % timeF == 0):
            frame1 = cv2.resize(frame, (128, 64))
            save_image(frame1, path + '/', j)
            j = j + 1
        success, frame = videoCapture.read()
        i = i + 1
    if max != 0:
        suc += 1
print(num)