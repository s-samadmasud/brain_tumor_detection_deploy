import streamlit as st  # For building web apps  
import numpy as np  # For numerical computations and array handling  
import cv2  # For image processing and computer vision tasks  
from PIL import Image  # For handling and manipulating images  
import tensorflow as tf  # For deep learning and neural networks  
from tensorflow import keras  # High-level API for building deep learning models  
from keras.layers import Dense  # For adding fully connected layers in neural networks  
from keras.models import Sequential, load_model  # For defining and loading deep learning models  

# Error handling for model loading (ensure the correct file format)
try:
  # Load pre-trained model (assuming it's saved as 'model.h5')
  model = load_model('model.h5')

  model_loaded = True  # Flag to indicate successful model loading
except OSError:
  st.error("Error: 'model.h5' file not found. Please ensure the model is saved in the correct location.")
  model_loaded = False  # Flag to indicate unsuccessful model loading
except ValueError as e:
  if "File format not supported" in str(e):
    st.error("Error: The model file format is not supported. Please ensure it's saved in HDF5 (.h5) format.")
  else:
    st.error(f"Error loading model: {e}")  # Handle other potential errors
  model_loaded = False  # Flag to indicate unsuccessful model loading

# Function to preprocess the image
def preprocess_image(image):
  """
  Preprocesses an image for use with the brain tumor detection model.

  Args:
      image: A NumPy array representing the image.

  Returns:
      A NumPy array representing the preprocessed image.
  """

  # Resize the image to the expected input shape
  image = cv2.resize(image, (128, 128))

  # Convert to RGB format (assuming model expects RGB)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Normalize pixel values
  image = image.astype('float32') / 255

  # Reshape for model prediction (add a batch dimension)
  image = np.expand_dims(image, axis=0)

  return image

# Streamlit application layout
st.title("Brain Tumor Detection")
st.write("Upload an image to classify it as normal or tumor-affected.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Add a prediction button only if a model is loaded
if uploaded_file is not None and model_loaded:
  predict_button = st.button("Predict")

  # Execute prediction logic only when button is clicked
  if predict_button:
    # Read the uploaded image
    image = np.array(Image.open(uploaded_file))

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])

    # Display the uploaded image
    st.image(image, caption="Uploaded Image")

    # Display prediction results based on predicted class
    if predicted_class == 1:
      st.error("Alert! You might have a Brain Tumor. Please consult a doctor for further evaluation.")
    else:
      st.success("Your brain scan appears normal. However, it's recommended to consult a doctor for regular checkups.")

  # Debugging (optional): Check button click state
  # st.write("Button clicked:", predict_button)

# Instructions for saving the model (optional)
st.write("**To save the model for future use:**")
# Provide instructions on how to save the model using the appropriate method (e.g., TensorFlow's `save_model` or Keras' `model.save`)
