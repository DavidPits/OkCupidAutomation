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
import random
import keras
import cv2

def load_dataset_train(batch_size):

    while(True):
        current_batch_clean=np.zeros((batch_size,150,150,3))
        current_batch_tags=np.zeros((batch_size,1))
        files_nl=os.listdir("face_nlike")
        files_n2=os.listdir("face_like")
        list_file_nl=random.sample(files_nl,batch_size//2)
        list_file_l=random.sample(files_n2,batch_size//2)
        cominbed=list_file_l+list_file_nl
        shuffled = random.sample(cominbed, len(cominbed))
        for i,pic in enumerate(shuffled):
            if pic[-5]=="1":
                current_batch_tags[i,:]=1
                pic="face_like/"+pic
            else:
                pic="face_nlike/"+pic
                current_batch_tags[i,:]=0
            current_batch_clean[i,:,:,:]=cv2.imread(pic)

        yield (current_batch_clean,current_batch_tags)


def load_dataset_test(batch_size):

    while(True):
        current_batch_clean=np.zeros((batch_size,150,150,3))
        current_batch_tags=np.zeros((batch_size))
        files_n2=os.listdir("test")
        list_file_nl=random.sample(files_n2,batch_size)
        for i,pic in enumerate(list_file_nl):
            if pic[-5]=="1":
                current_batch_tags[i]=1
            else:
                current_batch_tags[i]=0
            pic = "test/" + pic
            current_batch_clean[i,:,:,:]=cv2.imread(pic)

        yield (current_batch_tags,current_batch_clean)

def train_model(fit):


    model = define_net()

    gen_data=load_dataset_train(32)
    if fit==True:
        model.fit(gen_data,steps_per_epoch=5)
    else:
        model.load_weights("weights_curr_m")
    tags,imgs=next(load_dataset_test(50))
    predict_results(imgs, model, tags)


def predict_results(imgs, model, tags):
    predicts = model.predict(imgs)
    y_pred = [1 if (i > 0.5) else 0 for i in predicts]
    i = 0
    total = 0
    for predict, true in zip(y_pred, tags):
        print(predict, " ", true)
        if predict == true:
            i += 1
        total += 1
    print(i / total)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tags, y_pred, labels=[1, 0])
    model.save_weights("weights_curr_m")
    print(cm)


def define_net():
    resnet = ResNet50(include_top=False, pooling='avg')
    model = Sequential()
    model.add(resnet)
    model.add(Dense(1))
    model.add(Dropout(0.2))
    model.add(Activation('sigmoid'))
    model.layers[0].trainable = False
    ADAM = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=ADAM, metrics=["accuracy"])
    return model


def label_faces(path, x_train, y_train,tag):
    faces = os.listdir(path)
    i=0
    if tag==0:
        i=91
    for face in faces:
        img = cv2.imread(path + "/" + face)
        x_train[i,:,:,:]=img
        y_train[i]=tag
        i+=1

def main():

    train_model(False)
if __name__=="__main__":
    main()
