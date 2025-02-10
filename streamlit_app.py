import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model('path_to_your_trained_model.h5')

st.title("MNIST Digit Recognizer")

# Instructions for users
st.write("""
         This is a simple Streamlit app where you can draw a digit (28x28 pixels) and predict its class using a pre-trained MNIST model.
         Draw a digit in the canvas below and click the "Predict" button to see the result.
         """)

canvas_result = st.canvas(
    fill_color="white",  
    stroke_color="black",  
    stroke_width=30,  
    height=280,  
    width=280,  
)

if st.button('Predict'):

    if canvas_result.image_data is not None:
       
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
        img = img.convert('L')  
        img = img.resize((28, 28))
        img_array = np.array(img) 

        
        img_array = img_array / 255.0
        img_array = img_array.reshape(1,28*28)  
        
        
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

    
        st.image(img, caption="Your Drawing", use_column_width=True)
        st.write(f"Predicted Digit: {predicted_digit}")
    else:
        st.write("Please draw a digit to make a prediction.")

