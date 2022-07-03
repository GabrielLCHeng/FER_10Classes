import streamlit as st
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import numpy as np
# from typing import List, Union
import time
from PIL import Image
from fastai.vision.core import PILImage, to_image

import data
import model_utils
import haarcascade_detect

CASC_PATH = './media/haarcascade_frontalface_default.xml'
# CASC_PATH = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(CASC_PATH)

img_saving_path = './media/savedimages'
model_path = './media/models'       # .pth

def image_input(model_name):
    chosen_model = data.fer_model_dict[model_name]
    
    # image
    st.sidebar.write('')
    # upload image
    if st.sidebar.checkbox('Upload'):
        content_img = st.file_uploader("Upload human face image to predict emotion (png, jpg)", type=["png", "jpg", "jpeg"])
        if content_img is not None:
            input_img = Image.open(content_img)
            # # save input image
            # timestr = time.strftime("%Y%m%d-%H%M%S")
            # img_name = f'img_{timestr}'
            # input_img.save(f"{img_saving_path}/{img_name}.jpg")

            # need to convert to ndarray in order to predict
            content_img = np.array(input_img)
        else:
            st.warning('Upload an image OR untick the Upload checkbox')
            st.stop()
    # use provided image
    else:
        content_img_name = st.sidebar.selectbox("Choose an image to predict emotion", data.content_images_name)
        content_img = data.content_images_dict[content_img_name]
        # content_img = Image.open(content_img)

    # content_img = PILImage.create(content_img)
    # display image
    st.image(content_img, width=350)
    # do prediction
    emoClass, preds = model_utils.predict_from_frame(chosen_model, content_img)
    st.write('Predicted Emotion: ', emoClass)
    st.write('Probability: ', round(preds, 5))
    
    


def webcam_input(model_name):
    chosen_model = data.fer_model_dict[model_name]
    # chosen_model = data.fer_model_dict['Resnet34']
    # st.write(chosen_model)

    media_stream_constraints = {
    "video": {
        "width" : {"min": 800, "ideal": 1280, "max": 1920 },
        # "height": {"min": 480, "ideal":720, "max": 1080},
    },
    "audio": False
    }

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            in_img = frame.to_ndarray(format="bgr24")       # from h,w,c ->(360, 640, 3), can get higher

            img_gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
            # img_normal = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)        # type numpy.ndarray (360, 640, 3)   
            faces = cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=2)       # type: tuple
            for x,y,w,h in faces:   # 类似 319 136 129 129
                rec_img = cv2.rectangle(img=in_img, pt1=(x,y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)     # RGB green, return a numpt.ndarray with shape (360, 640, 3)
                cropped_image = in_img[y:y+h, x:x+w]

                emoClass, preds = model_utils.predict_from_frame(None, cropped_image)
                emoClass = str(emoClass)
                label_position = (x,y)
                cv2.putText(in_img, emoClass, label_position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(in_img, format='bgr24')
    

    stream = webrtc_streamer(
        key='gab_fer', 
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        # rtc_configuration=RTCConfiguration,
        media_stream_constraints=media_stream_constraints,
        video_processor_factory=VideoProcessor, 
        async_processing=True
    )
    