import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

import librosa 
import matplotlib.pyplot as plt


audio_path = r"/home/pi/Desktop/mbyr/predict_sed_audio/test.wav"

librosa_audio_data, librosa_sample_rate = librosa.load(audio_path)

plt.figure(figsize=(12,4))
plt.plot(librosa_audio_data)
plt.title('Audio Wavwform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()



def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features
    


# Load the trained model
model_1 = load_model(r"/home/pi/Desktop/mbyr/saved_models/audio_classification.hdf5")  

# Load the label encoder classes
labelencoder_classes_1 = np.load(r"/home/pi/Desktop/mbyr/saved_models/label_encoder_classes.npy")

# model_1.summary()

# Directory containing audio files for prediction
prediction_directory = r"/home/pi/Desktop/mbyr/predict_sed_audio"


# Iterate through all files in the directory
for filename in os.listdir(prediction_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(prediction_directory, filename)

        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

        # Reshape for prediction
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        # Make predictions using the loaded model
        predictions = model_1.predict(mfccs_scaled_features)

        # Get the index of the maximum value in predictions for each sample
        predicted_label = np.argmax(predictions, axis=1)
        print(predicted_label)
        
        # Convert the predicted label back to the original class using the label encoder classes
        prediction_class = labelencoder_classes_1[predicted_label[0]]

        print(f"File: {filename}, Predicted Class: {prediction_class}")
        print("-----------------------------")


