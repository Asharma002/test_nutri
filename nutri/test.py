

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import emoji
import keras

path = "./my_model2.hdf5"

model = keras.models.load_model(path)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """ 
    Preprocess the image to be compatible with the model.
    Args:
        image: Image to be processed.
    Returns:
        Processed image.
    """
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def app():
    st.title('NutriScore: A Deep Learning-based Food Classification System')
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    score = 0
    
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_array = preprocess_image(image)
        
        prediction = model.predict(image_array)
        
        class_names = ['Healthy', 'Unhealthy']
        prediction_class = class_names[np.argmax(prediction)]
        
        if prediction_class == "Unhealthy":
            score -= 1
            st.write('This image is Unhealthy.', emoji.emojize(":disappointed_face:"))
        elif prediction_class == "Healthy":
            score += 1
            st.write("Hello! :wave: This image is healthy. :smile:")
        else:
            print("Unexpected prediction class:", prediction_class)

    if score > 0:
        st.write('Yeah!! Final Score:', score)
        ss = emoji.emojize(":star-struck:")
        st.write(f'<span style="font-size: 3rem">{ss}</span>', unsafe_allow_html=True)
    elif score < 0:
        st.write('Final Score:', score)
        sob1 = emoji.emojize(":sob:")
        st.write(f'<span style="font-size: 3rem">{sob1}</span>', unsafe_allow_html=True)
    elif score == 0:
        st.write('Final Score:', score)
        nef = emoji.emojize(":neutral_face:")
        st.write(f'<span style="font-size: 3rem">{nef}</span>', unsafe_allow_html=True)

if __name__ == '__main__':
    app()
