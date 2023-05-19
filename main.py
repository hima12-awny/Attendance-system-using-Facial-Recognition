import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime

path_normal = 'images'
images = []
class_name = []
img_name = os.listdir(path_normal)

for cl in img_name:
    crImg = cv2.imread(f'{path_normal}\\{cl}')
    images.append(crImg)
    class_name.append(cl.split('.')[0])


def findEncode(images_list):

    encodedImgs = []
    for image in images_list:
        currant_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crEncode = face_recognition.face_encodings(currant_image)[0]
        encodedImgs.append(crEncode)

    return encodedImgs


def attendance(user_name):
    with open('attendance.csv', 'r+') as f:
        namelist = []
        for line in f.readlines():
            namelist.append(line.split(',')[0])

        if user_name not in namelist:
            f.write(f'\n{user_name},{datetime.now().strftime("%H:%M:%S")}')


print('Encoding...')
encodeListKnown = findEncode(images)
print('Encoding Complete.')
print('Wait For Opening Your WebCam..')

cap = cv2.VideoCapture(0)
cap.set(3, 700)
cap.set(4, 700)
print('our WebCam is ready now.'.title())

while 1:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (0, 0), None, 1.5, 1.5)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgRGB)
    encodeCurFrame = face_recognition.face_encodings(imgRGB, facesCurFrame)

    for encodeFace, face_location in zip(encodeCurFrame, facesCurFrame):

        match = face_recognition.compare_faces(encodeListKnown, encodeFace)

        dis = face_recognition.face_distance(encodeListKnown, encodeFace)

        nameIdx = np.argmin(dis)
        y1, x2, y2, x1 = face_location

        if match[nameIdx]:
            name = class_name[nameIdx].upper()

            attendance(name)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (255, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'NO MATCH', (x1 + 6, y2 - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()