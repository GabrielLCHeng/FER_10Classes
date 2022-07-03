import streamlit as st
import data
from input import image_input, webcam_input
# from haarcascade_detect import *


def main():
    st.markdown("<h1>Facial Emotion Recognition App</h1>", unsafe_allow_html=True)

    # initialize session state
    if "user" not in st.session_state:
        # Will store the currently logged user
        st.session_state.user = None
    # print(st.session_state.user)
    st.caption('10 Emotion Classes: Neutral, Happiness, Sadness, Anger, Fear, Surprised, Disgust, Confused, Thinking, Boredom')
    # navigation
    st.sidebar.title('Navigation')
    method = st.sidebar.radio('Go To \u2192', options=['Webcam', 'Image'])
    # pick models
    st.sidebar.header('Options')
    modelsOption = st.sidebar.selectbox('Choose model to use', data.fer_model_dict.keys())
    st.sidebar.write(modelsOption, 'is selected.')
    # print(modelsOption)

    if method == 'Image': # image mode
        image_input(modelsOption)   
    else: # webcam mode
        webcam_input(modelsOption)


main()