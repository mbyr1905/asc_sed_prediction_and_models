import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

import librosa 
import matplotlib.pyplot as plt
from tensorflow.lite.python.interpreter import Interpreter

# audio_path = r"/home/pi/Desktop/mbyr/predict_sed_audio/test.wav"

# librosa_audio_data, librosa_sample_rate = librosa.load(audio_path)

# plt.figure(figsize=(12,4))
# plt.plot(librosa_audio_data)
# plt.title('Audio Waveform')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.show()

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path=r"D:\prediction\saved_models_tflite\converted_model_sed.tflite")
interpreter.allocate_tensors()

# Load the label encoder classes
labelencoder_classes = np.load(r"D:\prediction\saved_models_tf\label_encoder_classes.npy")

# Directory containing audio files for prediction
prediction_directory = r"D:\audio classification\UrbanSound8K\predict_sed"

# Iterate through all files in the directory
for filename in os.listdir(prediction_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(prediction_directory, filename)

        # Extract features
        mfccs_scaled_features = features_extractor(file_path)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        # Prepare input tensor for the model
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], mfccs_scaled_features.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_details = interpreter.get_output_details()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted class label
        predicted_label = np.argmax(output, axis=1)[0]
        prediction_class = labelencoder_classes[predicted_label]

        print(f"File: {filename}, Predicted Class: {prediction_class}")
        print("-----------------------------")
