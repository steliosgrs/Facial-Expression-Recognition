import numpy as np
import cv2
import streamlit as st
import tensorflow
from tensorflow import keras

from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# Load model
emotion_dict = {0:'angry', 1 :'disgust', 2: 'fear', 3:'happy', 4: 'neutral', 5:'sad' , 6:'surprise'}
classifier = load_model('D:cnn_model200_32.h5')

# Load face
try:
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h),
                          color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                percent = int(prediction[prediction.argmax()] * 100)  # New
                percent = str(percent) + '%'
                percent_position = (x + w, y - 10)  # New
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(percent), percent_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # New

        return img

def main():
    # Face Analysis Application #
    st.title("Εφαρμογή για την διπλωματική εργασία \"Αυτόματη Ανίχνευση, Ανάλυση και Αναγνώριση Συναισθημάτων\"")
    activiteis = ["Home","Analyze Image Emotion", "Webcam Emotion Recognition", "About"]
    choice = st.sidebar.selectbox("Επιλογή Ενέργειας", activiteis)
    st.sidebar.markdown(
        """ """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Εφαρμογή Αναγνώρισης Συναισθημάτων Προσώπου μέσω CNN</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 Η εφαρμογή έχει 2 λειτουργίες.
                 1. Αναγνώριση Συναισθήματος σε πραγματικό χρόνο μέσω κάμερας.
                 2. Ανάλυση Συναισθήματος σε φωτογραφία.
                 """)
    elif choice == "Webcam Emotion Recognition":
        st.header("Webcam Real-time Recognition")
        # st.write("Click on start to use webcam and detect your face emotion")
        html_temp_Webcam1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Εφαρμογή αναγνώρισης συναισθήματος στο πρόσωπο</h4>
                                    <br>
                                    <strong>Βήματα για την λειτουργία:</strong>
                                    <ol type = "1">
                                    <li>Επιλογή συσκευή κάμερας</li>
                                    <li>Εκίνηση της κάμερας</li>
                                    </ol> 
                                    </div>
                                    </br>"""
        st.markdown(html_temp_Webcam1, unsafe_allow_html=True)

        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
        col1,col2 = st.columns(2)
        with col1:
            col1.header("Text")
            col1.write("POU EISAI")


    elif choice == "Analyze Image Emotion":
        st.header("Webcam Live Feed")
        st.write("Click to upload your image and ")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("Λίγα λόγια για την εφαρμογή")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Εφαρμογή αναγνώρισης συναισθήματος στη περιοχή του προσώπου.</h4>
                                    <br>
                                    <strong>Χρησιμοποιήθηκαν:</strong>
                                    <ol type = "1">
                                    <li>Το εκπαιδευμένο μοντέλο CNN για την αναγνώριση.</li>
                                    <li>Η OpenCV βιβλιοθήκη για την λειτουργία της κάμερας σε πραγματικό χρόνο.</li>
                                    <li>Το framework του streamlit για την δημιουργία της Web εφαρμογής.</li>
                                    </ol> 
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        # <h5 style="color:white;text-align:center;">Ευχαριστώ </h5>
        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">Η εφαρμογή δημιουργήθηκε από τον Στέλιο Γεωργαρά</h4>
                             		
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()