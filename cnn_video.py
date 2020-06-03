from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
from os import listdir
import numpy as np
from tensorflow.keras.models import load_model

from mtcnn.mtcnn import MTCNN
label_names = {}
forder = "dataset/raw"
index_label = 0
for forder_name in listdir(forder):
    label_names[index_label] = forder_name 
    index_label += 1
print(label_names)   


mode = load_model("model1.h5")
mode.load_weights("model_weights1.h5")
detector = MTCNN()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face = detector.detect_faces(frame)
        print(len(face))
        for person in face:
            bounding_box = person['box']
            keypoints = person['keypoints']
            im_crop = frame[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                # cv2.imwrite(os.path.join("data/raw/" + raw_name  , filename), im_crop)
                im_crop = tf.convert_to_tensor(im_crop)
                im_crop = tf.cast(im_crop, tf.float32)
                im_crop = (im_crop/255)   
                print(im_crop.shape)        #normalize
                im_crop = tf.image.resize(im_crop, (256, 256))
                im_crop = [im_crop]
                im_crop = tf.convert_to_tensor(im_crop)
                # im_crop = tf.reshape()
                prediction = mode.predict(im_crop)
                print(prediction)
                max_index = int(np.argmax(prediction))
                # print()
                # print(max_index)
                # print(label_names[max_index])

                cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
                # cv2.putText(frame, label_names[max_index], (bounding_box[0]+20, bounding_box[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, label_names[max_index], (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)
            # cv2.imwrite(path + "abc.jpg",image)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    else:
        break

cap.release()
cv2.destroyAllWindows()




