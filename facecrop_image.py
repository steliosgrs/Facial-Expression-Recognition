import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

directory = 'C:/Users/NoLifer/Facial-Expression-Recognition/images/'
results_dir = 'C:/Users/NoLifer/Facial-Expression-Recognition/cropped/'
# face_roi = np.zeros((750,750))
def facecrop(image,name):

    frame = cv2.imread(image)
    # Load the cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    face_roi = cv2.imread(image)
    img = cv2.imread(image,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    faces = cascade.detectMultiScale(img,1.1,4)
    for x,y,w,h in faces:
        roi_gray = img[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame,(x,y), (x+w , y+h), (255,0,0), 2)
        facess = cascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print('Face not detected\n')
        else:
            for (ex, ey, ew, eh) in facess:
                # Crop the face
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]
                dir =os.path.join(results_dir, f"Cropped {name}")
                print(dir)
                print(face_roi)
                cv2.imwrite(dir, face_roi)

                # print(type(face_roi))

    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    return  face_roi

def getCroppedImages(images):
    return images

if __name__ == '__main__':
    images = os.listdir(directory)
    i = 1
    only_faces = []
    for img in images:

        file = directory + img
        # print(i)
        image = cv2.imread(file)
        print(img)
        plt.subplot(2, len(images), i)

        temp = facecrop(file,img)

        plt.subplot(2, len(images), i + len(images))
        plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        only_faces.append(temp)

        i += 1

    getCroppedImages(only_faces)
    # print(only_faces[1])
    # cv2.imshow(only_faces[1],cv2.COLOR_BGR2RGB)
    # plt.imshow(cv2.cvtColor(only_faces[3], cv2.COLOR_BGR2RGB))
