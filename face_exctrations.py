import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import mtcnn
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from keras.applications import ResNet50
from keras import *
from keras.layers import *

import cv2

NOT_LIKED = False

LIKED = True

FREQUNICES = 3
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from matplotlib.patches import Rectangle

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def get_file_names():
    dir="train"
    path=dir+"/"

    i=0
    for name in os.listdir(dir):
        print(name)
        if name[-5]=="2":
            os.rename(path+name,path+name[:-6]+"2.png")
        i+=1
def detect_faces():
    all_file_names = os.listdir("liked")
    succ = 0
    total = 0
    check_if_low = []
    for file in all_file_names:
        current_path = "liked/" + file
        tof, mean = detect_face(current_path, file)
        if tof == True:
            check_if_low.append((mean, file))
            succ += 1
        total += 1
    check_if_low.sort()


def detect_face(img_file_name, only_ending):
    img = cv2.imread(img_file_name, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(equalized_img, 1.1, minNeighbors=5)
    equalized_img = np.array(equalized_img)

    for (x, y, w, h) in faces:
        current_face = equalized_img[y:y + 15 + h, x:x + 15 + w]
        eyes = eye_cascade.detectMultiScale(current_face, scaleFactor=1.1, minNeighbors=3)
        low_freq_mean = get_low_frequcnies_mean(current_face)
        i = 0
        if (len(eyes) == 0) and low_freq_mean > 30:
            i += 1
            print(img_file_name, low_freq_mean)
            return False, False
        cv2.imwrite("faces_like/" + only_ending, current_face)
        return True, low_freq_mean
    return False, False


def get_low_frequcnies_mean(current_face):
    """
    Trying to get the right amount of normazilation to detect which pics that dont have eyes at all in them also deosn't include
    a face by  looking at the mean of their low frequncies.
    :param current_face:
    :return:
    """
    f = np.fft.fft2(current_face)
    fshift = np.fft.fftshift(f)
    fshift = 20 * np.log(np.abs(fshift))
    mid_point_x = current_face.shape[0] // 2
    mid_point_y = current_face.shape[1] // 2
    fshift = fshift / fshift.sum()
    fshift *= 100000
    low_freq_mean = fshift[mid_point_x:mid_point_x + FREQUNICES, :mid_point_y:mid_point_y + FREQUNICES].mean()
    return low_freq_mean


# draw each face separately
def draw_faces(filename, result_list):
    # load the image
    data = cv2.imread(filename,1)
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        pyplot.subplot(1, len(result_list), i + 1)
        pyplot.axis('off')
        # plot face
        pyplot.imshow(data[y1:y2, x1:x2])
    # show the plot
    pyplot.show()




def MTCNN_face_detection():
    final_path="pre_proccesed_pics/"
    path="pre_proccesed_pics"
    all_file_names = os.listdir("pre_proccesed_pics")
    detector = MTCNN()
    extract_face_NN(all_file_names, detector, final_path, path)

def choose_path(liked_or_not=False):
    if liked_or_not==1:
        path= "current_attemp"
        final_path="current_attemp/"
        return final_path,path
    if liked_or_not == True:
        path = "liked"
        final_path = "face_like/"
    else:
        path = "passed"
        final_path = "face_nlike/"
    return final_path, path


def extract_face_NN(all_file_names, detector, final_path, path):
    for file in all_file_names:
        orig = file
        file = path + "/" + file
        pixels = cv2.imread(file, 1,)
        print(type(pixels))
        if type(pixels)!=np.ndarray:
            continue
        results = detector.detect_faces(pixels)
        if len(results) == 1:
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            if face.size < 250*250:
                os.remove(file)
                continue
            cv2.imwrite("faces_only/"+orig,face)
        os.remove(file)


def resizing_images(likeOrNot:bool):
    final_path,path=choose_path(likeOrNot)
    final_path="test"
    all_file_names = os.listdir(final_path)
    for file in all_file_names:
        img=cv2.imread(final_path+"/"+file,1)
        reszied=cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)
        cv2.imwrite(final_path+"/"+file,reszied)


def resizing_images_detect(final_path):
    all_file_names = os.listdir(final_path)
    for file in all_file_names:
        img=cv2.imread(final_path+"/"+file,1)
        if  img.size>250*250:
            rating=file[-5]
            reszied=cv2.resize(img,(250,250),interpolation=cv2.INTER_AREA)
            cv2.imwrite(rating+"_stars"+"/"+file,reszied)
            os.remove(final_path+"/"+file)

def move_entities_to_rating_fold():
    for ent in os.listdir("train"):
        rating = ent[-5]
        if rating=='p':
            continue
        try:
            os.rename("train/" + ent, rating + "_stars/" + ent)
        except:
            continue

if __name__=="__main__":

    move_entities_to_rating_fold()
    #te
    # MTCNN_face_detection()
    # resizing_images_detect("faces_only")