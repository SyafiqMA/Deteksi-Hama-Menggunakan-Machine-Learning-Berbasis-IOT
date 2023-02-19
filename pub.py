import random
import time

from paho.mqtt import client as mqtt_client
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

import tensorflow as tf

from PIL import Image
import cv2

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.ensemble import GradientBoostingClassifier
import xgboost
import warnings
import pickle

warnings.filterwarnings("ignore")

   

def predict():
    model3 = pickle.load(open('Adaboost-kfold.sav', 'rb'))
    TESTING_DIR = os.path.join('test')
    IMG_SIZE = (224, 224)
    IMG_SHAPE = IMG_SIZE + (3,)

    base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
    
    test_images = []
    test_labels = []


    for img_path in glob.glob('./hasil/*'):
        # print(img_path)
        # label = directory_path.split("/")[-1]
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        # test_labels.append(label)
        # label = ['test\Worms', 'test\healthy']
        # print(label)
        # for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):


    test_images = np.array(test_images)
    # test_labels = np.array(test_labels)
    
    
    le = preprocessing.LabelEncoder()

    
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    
    CLASSES = []

    folders = os.listdir(TESTING_DIR)
    for f in folders:
        CLASSES.append(f)
    
    test_features = base_model.predict(test_images)
    test_features = test_features.reshape(test_features.shape[0], -1)


    predictions3 = model3.predict(test_features)
    
    return predictions3

broker = 'broker.emqx.io'
port = 1883
topic1 = "detect/maize"
# topic0 = "detect/healthy"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'emqx'
password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client):
    
    msg_count = 0
    msg = ''
    while True:
        time.sleep(1)
        pred = predict()
        print (pred)
        deteksi = []
        for i in pred:
            if (i == 0):
                j ='Worms'
                deteksi.append(j)
            else:
                j = 'healthy'
                deteksi.append(j)
        print (deteksi)
    
        for x in deteksi:
            if (x == 'Worms'):
                msg = "Worms"
                result = client.publish(topic1, msg)
                status = result[0]
                if status == 0:
                    print(f"Send `{msg}` to topic `{topic1}`")
                else:
                    print(f"Failed to send message to topic {topic1}")
            # elif(x=='hama'):
            #     msg = "hama"
            #     print(f"Send `{msg}` to topic `{topic1}`")
            #     result = client.publish(topic1,msg)
            #     status = result[0]
            #     if status == 0:
            #         print(f"Send `{msg}` to topic `{topic1}`")
            #     else:
            #         print(f"Failed to send message to topic {topic1}")



def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)


if __name__ == '__main__':
    run()
