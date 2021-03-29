import cv2
import numpy as np
import os


def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)

def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

def get_image(id):
    try:
        image = cv2.resize(cv2.imread(path), (128, 64))
    except:
        return [], -1
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)

    dirs = os.listdir(path)
    max = 0
    tmp = 1000000
    frames = []
    for pic in dirs:
        frame1 = cv2.imread(path)
        frames.append(frame1.tolist())
        gray2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hash2 = getHash(gray2)
        ret = Hamming_distance(hash1, hash2)
        if ret < tmp:
            tmp = ret
            max = int(pic.split('.')[0])
        if len(frames) >= 200:
            break
    return frames, max

pic_features, cover_id = get_image('id')