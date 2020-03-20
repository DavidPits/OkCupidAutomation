import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from keras.applications import ResNet50
from keras import *
from keras.layers import *
import random
import keras
import cv2

def load_dataset_train(batch_size):
    while(True):
        current_batch_clean=np.zeros((batch_size,250,250,3))
        current_batch_tags=np.zeros((batch_size,1))
        stars_1_files=os.listdir("/content/1_stars")
        stars_2_files=os.listdir("/content/2_stars")
        stars_3_files=os.listdir("/content/3_stars")
        stars_4_files=os.listdir("/content/4_stars")
        stars_5_files=os.listdir("/content/5_stars")
        zero_class=stars_1_files+stars_2_files
        one_class=stars_4_files+stars_5_files+stars_3_files
        current_batch=random.sample(one_class,batch_size//2)+random.sample(zero_class,batch_size//2)

        shuffled = random.sample(current_batch, (batch_size))
        for i,pic in enumerate(shuffled):
            rating=pic[-5]
            assign_pic_n_tag(current_batch_clean,current_batch_tags,i,pic,rating+"_stars/")

        yield (current_batch_clean,current_batch_tags)


def load_dataset_test(batch_size):

    while(True):
        current_batch_clean=np.zeros((batch_size,250,250,3))
        current_batch_tags=np.zeros((batch_size))
        files_n2=os.listdir("test")
        list_file_nl=random.sample(files_n2,batch_size)
        for i,pic in enumerate(list_file_nl):
            assign_pic_n_tag(current_batch_clean, current_batch_tags, i, pic,"test/")

        yield (current_batch_tags,current_batch_clean)


def assign_pic_n_tag(current_batch_clean, current_batch_tags, i, pic,path):
    if pic[-5] == "5":
        current_batch_tags[i] = 1
        pic = path + pic
    if pic[-5] == "4":
        current_batch_tags[i] = 1
        pic = path + pic
    if pic[-5] == "3":
        current_batch_tags[i] = 1
        pic = path + pic
    if pic[-5] == "2":
        pic = path + pic
        current_batch_tags[i] = 0
    if pic[-5] == "1":
        pic = path + pic
        current_batch_tags[i] = 0
    current_batch_clean[i, :, :, :] = cv2.imread(pic)


def train_model(fit):


    model = define_net()

    gen_data=load_dataset_train(16)
    if fit==True:
        model.fit(gen_data,steps_per_epoch=15,epochs=30)
    else:
        model.load_weights("weights_curr_m")
    tags,imgs=next(load_dataset_test(32))
    predict_results(imgs, model, tags)


def predict_results(imgs, model, tags):
    predicts = model.predict(imgs)
    total=0
    current=0
    y_pred = [1 if (i > 0.5) else 0 for i in predicts]

    for predict, true in zip(y_pred, tags):
        print(predict, " ", true)
        if predict==true:
          current+=1
        total+=1
    print(current/total)

    model.save_weights("weights_curr_m")


def define_net(drop_rate=0.5):

    model = applications.VGG19(weights= "imagenet", include_top = False, input_shape = (250, 250, 3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    new_model = Sequential()  # new model
    for layer in model.layers:
        new_model.add(layer)

    new_model.add(top_model)  # now this works
    for layer in model.layers[:21]:
        layer.trainable = False
    new_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return new_model



def main():

    train_model(True)
if __name__=="__main__":
    main()
