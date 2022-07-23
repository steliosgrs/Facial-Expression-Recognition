import numpy as np
import cv2
import streamlit as st
import tensorflow
from tensorflow import keras
import time
import threading as th
import queue
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode, ClientSettings
import av
from multiprocessing import Queue, Process
import copy
# import
# from utils import CvFpsCalc
# from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx
from streamlit.scriptrunner import add_script_run_ctx as add_report_ctx

st.set_page_config(layout="wide")

# Load model
emotion_dict = {0:'angry', 1 :'disgust', 2: 'fear', 3:'happy', 4: 'neutral', 5:'sad' , 6:'surprise'}
classifier = load_model('D:cnn_model200_32.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
Current_emotion = []
# thread = th.Thread(target=timer,daemon=True)
# add_report_ctx(thread)
# t = thread.start()
last5frames = []

# Load face
try:
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Cascade for face detection loaded")
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def Emotion_detection(state):

    col1, col2 = st.columns(2, gap='small')

    class FaceEmotion(VideoProcessorBase):
        def __init__(self) -> None:
            self.current_emotion = ""
            # return self.current_emotion

        def Face_Emotion(self, frame):
            output=""
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            labels = []
            # image gray
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
                    label = emotion_labels[prediction.argmax()]


                    # if len(last5frames) < 5:
                    #     last5frames.append(label)  # New
                    # elif len(last5frames) == 5:
                    #     print(last5frames)
                    #     count = last5frames
                    #     last5frames.pop(0)
                    # else:
                    #     last5frames.pop(0)  # New
                    # with lock:
                    #     Current_emotion = most_frequent(last5frames)
                    #
                    # if t == 0:
                    #     choices(Current_emotion)
                    #     t = 5

                    # Print the percent and the % in position
                    percent = int(prediction[prediction.argmax()] * 100)  # New
                    percent = str(percent) + '%'
                    percent_position = (x + w, y - 10)  # New

                    # Get Current dominant emotion
                    maxindex = int(np.argmax(prediction))
                    finalout = emotion_dict[maxindex]
                    output = str(finalout)

                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(percent), percent_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # New

            return img, output

        def recv(self, frame):
            # img = frame.to_ndarray(format="bgr24")
            # img = cv2.flip(img, 1)
            # image = copy.deepcopy(img)
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img, result = self.Face_Emotion(frame)

            if result == "":
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            else:
                self.current_emotion = result[0]
                return  av.VideoFrame.from_ndarray(img, format="bgr24")

            # return frame_out

    stream = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                                 media_stream_constraints={"video": True, "audio": False},
                                 desired_playing_state=state,video_processor_factory=FaceEmotion,
                                 async_processing=True)

    # while True:
    #     time.sleep(0.1)
    #     if stream.state.playing:
    #         emotion = stream.video_processor.current_emotion
    #         print("FOUND")
    #         col1.subheader(emotion)
    #         del stream

    # return FaceEmotion(VideoProcessorBase)

def main():
    st.title("Αυτόματη Ανίχνευση, Ανάλυση και Αναγνώριση Συναισθημάτων")
    activiteis = ["Home", "Analyze Image Emotion", "Webcam Emotion Recognition", "About"]
    choice = st.sidebar.selectbox("Επιλογή Ενέργειας", activiteis)
    st.sidebar.markdown(""" """)
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
        st.header("Webcam Real-time Emotion Recognition")
        html_temp_Webcam1 = """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Εφαρμογή αναγνώρισης συναισθήματος στο πρόσωπο</h4>
                                <br>
                                <strong>Βήματα για την λειτουργία:</strong>
                                <ol type = "1">
                                <li>Επιλογή συσκευή κάμερας</li>
                                <li>Εκίνηση της κάμερας</li>
                                </ol></div></br>"""
        st.markdown(html_temp_Webcam1, unsafe_allow_html=True)
        state = True
        possible_emotion = Emotion_detection(state)
        # col1, col2 = st.columns(2, gap='small')
        #
        # with col1:
        # display_mode = st.radio("Display mode", ["Emotion", "Face", "Both"], index=0)
        #
        # def processor_factory():
        #     return Faceemotion(
        #         static_image_mode=static_image_mode,
        #         display_mode=display_mode
        #     )

        # webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
        #                              media_stream_constraints={"video": True, "audio": False},
        #                              video_processor_factory=processor_factory, async_processing=True)

        # st.session_state["started"] = webrtc_ctx.state.playing
        # if webrtc_ctx.video_processor:
        #     webrtc_ctx.video_processor.rev_color = rev_color
        #     webrtc_ctx.video_processor.display_mode = display_mode

        # while webrtc_ctx.state.playing:
        #     with lock:
        #         Current_emotion = most_frequent(last5frames)

        # if st.checkbox("Show the detected labels", value=True):
        #     if webrtc_ctx.state.playing:
        #         labels_placeholder = st.empty()
        #         # NOTE: The video transformation with object detection and
        #         # this loop displaying the result labels are running
        #         # in different threads asynchronously.
        #         # Then the rendered video frames and the labels displayed here
        #         # are not strictly synchronized.
        #         while True:
        #             if webrtc_ctx.video_processor:
        #                 try:
        #                     result = webrtc_ctx.video_processor.result_queue.get(
        #                         timeout=1.0
        #                     )
        #                 except queue.Empty:
        #                     result = None
        #                 labels_placeholder.table(result)
        #             else:
        #                 break

        # with col2:
        #     col2.header("Text")
        #
        #     html_column2 = """<div style="background-color:#6D7B8D;padding:10px">
        #                     <h4 style="color:white;">POU EISAI2</h4>
        #                     </div>
        #                     </br>"""
        #     st.markdown(html_column2, unsafe_allow_html=True)

    elif choice == "Analyze Image Emotion":
        st.header("Webcam Live Feed")
        st.write("Click to upload your image and ")
        ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                              video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("Λίγα λόγια για την εφαρμογή")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
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