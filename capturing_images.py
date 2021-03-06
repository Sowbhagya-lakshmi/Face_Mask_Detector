import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
pause = True

# Asking the type of img from the user
while True:
    global category

    type_of_img = input('Do want to capture mask or no_mask images ? (m/n): ')
    
    if type_of_img == 'm':
        category = 'mask'
        break
    elif type_of_img == 'n':
        category = 'no_mask'
        break
    else:
        print('Enter valid letters please !')

# To determine from which number(in the name of jpg imgs) the saving of image should start
full_path = 'face_images/'+category+'/'

num_list = []
list_of_all_images = os.listdir(full_path)
try:
    for img_str in list_of_all_images:
        splitted_list = img_str.split('.')  # splitting the list at '.' and extracting the numbers before '.jpg'
        num_list.append(int(splitted_list[0]))
    last_num = sorted(num_list)[-1]  # gives the number of the last one in ascending order
    count = last_num + 1
except:
    count = 0

# MAIN ALGORITHM
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

initial_count = count
while True:
    ret, frame = video.read()
    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x:x+w]
        cv2.imshow('roi', roi)
        if not pause:
            count += 1
            print(count, 'Saving...')
            cv2.imwrite(full_path+str(count)+'.jpg', roi)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        break
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(15)
    if key == 27: break      # press 'esc' to finish
    if key == ord('p'):      # press 'p' to pause
        pause = not(pause)
    
    if count == initial_count + 30: break

video.release()
cv2.destroyAllWindows()