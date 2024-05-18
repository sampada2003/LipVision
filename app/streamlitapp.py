# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils_test import load_data_test, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('image.png')
    st.title('LipVision')
    st.info('An interactive web application that uses visual sequence to anticipate spoken phrases')

#st.title('LipVision Full Stack App') 
# Generating a list of options or videos 
st.info('Remove the uploaded video before selecting a video from the "Choose Video" dropdown!')
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)
upload_directory = '../data/s1'

uploaded_video = st.file_uploader("Upload a video", type=["mpg"])

if uploaded_video is not None:

    # Create a file path to save the uploaded video
    original_file_path = os.path.join(upload_directory, uploaded_video.name)

    # Save the uploaded video to the specified directory
    with open(original_file_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    selected_video = uploaded_video.name

    # Generate two columns 
    col1, col2 = st.columns(2)

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video = load_data_test(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif',width=380) 
        video = tf.expand_dims(video, axis=0)  # Add batch dimension
        video = tf.expand_dims(video, axis=-1)
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(video)
        #decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        decoder = tf.keras.backend.ctc_decode(yhat, input_length=np.ones(yhat.shape[0])*yhat.shape[1], greedy=True)[0][0].numpy()

        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

else:
    # Generate two columns 
    col1, col2 = st.columns(2)

    if options: 

        # Rendering the video 
        with col1: 
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join('..','data','s1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            # Rendering inside of the app
            video = open('test_video.mp4', 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)


        with col2: 
            st.info('This is all the machine learning model sees when making a prediction')
            video = load_data_test(tf.convert_to_tensor(file_path))
            imageio.mimsave('animation.gif', video, fps=10)
            st.image('animation.gif', width=377) 

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)