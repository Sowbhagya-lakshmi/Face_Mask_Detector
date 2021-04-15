import numpy as np
import cv2
import random
from tensorflow import keras

# Loading the trained model
model = keras.models.load_model('models/mask_no_mask_model.hdf5')

# Loading haar cascade xml
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def predict(img):
    prediction = model.predict_classes(img.reshape(1,50,50,3))
    if prediction == [[0]]:     #mask
        colour = (0,255,0)
        text = 'MASK DETECTED'
        print(text)
    elif prediction == [[1]]: # no mask
        colour = (0,0,255)
        text = 'NO MASK'
        print(text)
    return colour, text

def draw_stuff_on_frame(frame, colour, text):
    cv2.rectangle(frame, (x,y), (x+w, y+h), colour, 4)  # for bounding box

    # Putting text
    cv2.rectangle(frame,(x,y-30), (x+w, y), colour, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,text,(x+15,y-10), font, 0.5,(255,255,255),2,cv2.LINE_AA) 

################ MAIN ALGORITHM ################
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = video.read()
    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(roi, (50,50))
        normalised_img = resized_img/255
        # Predicting
        colour, text = predict(normalised_img)
        # Visualising the result
        draw_stuff_on_frame(frame, colour, text)
        break

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(25)
    if key == 27: break     # press 'esc' to finish

video.release()
cv2.destroyAllWindows()

















'''
dataset_path = 'dataset_sow/mask_no_mask_dataset.npy'
dataset = np.load(dataset_path, allow_pickle=True)

random.shuffle(dataset)

classes = ['mask, no_mask']

test_inputs = []
test_targets = []

for img, target in dataset[601:]:
    test_inputs.append(img)
    test_targets.append(target)

test_inputs = np.array(test_inputs)
test_targets = np.array(test_targets)

normalised_test_inputs = test_inputs/255

for index, test_input in enumerate(normalised_test_inputs):
    prediction = model.predict_classes(test_input.reshape(1,50,50,3))
    cv2.imshow('test_input', test_input)
    print(prediction)
    #print('target: ', classes[test_targets[index]], 'prediction: ', prediction)
    if cv2.waitKey(0) == 27: break

model.evaluate(normalised_test_inputs, test_targets)'''