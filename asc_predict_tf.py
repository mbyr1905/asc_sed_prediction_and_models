from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os
import wave
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

# Load the trained model
model = load_model(r"D:\prediction\saved_models\model_asc.hdf5")  # Provide the correct path to your model file

# Load the label encoder classes
label_encoder_classes = np.load(r"D:\prediction\saved_models\label_encoder_classes_asc.npy")

# model.summary()

# Directory containing audio files for prediction
prediction_directory = r"D:\audio classification\UrbanSound8K\predict_asc"
OUTPUT_DIR = r"D:\prediction\saved_params"
OUTPUT_DIR_ZCR = r"D:\prediction\saved_params\zcr"
OUTPUT_DIR_MFCC= r"D:\prediction\saved_params\mfcc"
OUTPUT_DIR_LOG_MEL=r"D:\prediction\saved_params\audio-images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_ZCR, exist_ok=True)
os.makedirs(OUTPUT_DIR_MFCC, exist_ok=True)
os.makedirs(OUTPUT_DIR_LOG_MEL, exist_ok=True)

# Function to calculate ZCR and save results
def calculate_zcr_and_save(audio_file, output_dir):
    try:
        # Read the WAV file
        wav = wave.open(audio_file, 'r')
        frames = wav.readframes(-1)
        sound_info = np.frombuffer(frames, dtype=np.int16)
        frame_rate = wav.getframerate()
        wav.close()

        # Calculate ZCR
        zcr = np.mean(np.abs(np.diff(np.sign(sound_info))) / 2.0)

        # Extract the file name without extension
        filename = os.path.splitext(os.path.basename(audio_file))[0]

        # Save the ZCR value to a text file in the output directory
        zcr_file_path = os.path.join(output_dir, 'test_zcr.txt')
        with open(zcr_file_path, 'w') as f:
            f.write(f'ZCR: {zcr}')
    except Exception as e:
        print(f"An error occurred for {audio_file}: {e}")
        

# Function to calculate MFCCs and save results
def calculate_mfcc_and_save(audio_file, output_dir):
    try:
        # Extract the file name without extension
        filename = os.path.splitext(os.path.basename(audio_file))[0]

        # Calculate MFCCs
        audio, sample_rate = librosa.load(audio_file, sr=None)  # Load the audio file
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)  # You can adjust the number of MFCC coefficients
        # print(mfccs.shape)
        # Save the MFCCs to a binary file in the output directory (overwrite if it already exists)
        np.save(os.path.join(output_dir, 'test_mfcc.npy'), mfccs)
    except Exception as e:
        print(f"An error occurred for {audio_file}: {e}")
        

# Function to generate mel spectrogram and save as an image
def generate_mel_spectrogram_and_save(file_path, output_dir):
    try:
        file_stem = Path(file_path).stem
        file_dist_path = os.path.join(output_dir, 'test_image.png')

        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Calculate the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        # Convert to decibels
        log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Plot and save the spectrogram image (overwrite if it already exists)
        plt.figure(figsize=(8, 6))
        librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectrogram for {os.path.basename(file_path)}")
        plt.savefig(file_dist_path)
        plt.close()
    except Exception as e:
        print(f"An error occurred for {file_path}: {e}")


# Iterate through all files in the directory
for filename in os.listdir(prediction_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(prediction_directory, filename)

        # Assuming you have these functions defined
        calculate_zcr_and_save(file_path, OUTPUT_DIR_ZCR)
        calculate_mfcc_and_save(file_path, OUTPUT_DIR_MFCC)
        generate_mel_spectrogram_and_save(file_path, OUTPUT_DIR_LOG_MEL)

        # Load the processed data
        mfcc_data = np.load('saved_params/mfcc/test_mfcc.npy')
        zcr_file_path = 'saved_params/zcr/test_zcr.txt'

        # Load the spectrogram image and resize it to (64, 64)
        spectrogram_image_path = 'saved_params/audio-images/test_image.png'
        spectrogram_image = load_img(spectrogram_image_path, target_size=(64, 64))
        spectrogram_data = img_to_array(spectrogram_image)
        spectrogram_data = np.expand_dims(spectrogram_data, axis=0)  # Add batch dimension
        spectrogram_data = spectrogram_data / 255.0  # Normalize pixel values
        
        # Ensure that both data arrays have the same number of samples
        num_frames = 13  # Change this to match your model input size
        num_mfcc_coefficients = 862  # Change this to match your model input size
        mfcc_data = mfcc_data[:num_frames, :num_mfcc_coefficients]
        mfcc_data = mfcc_data.T
        mfcc_data = np.expand_dims(mfcc_data, axis=0)
        
        # Normalize the MFCC data (adjust based on your training data)
        # Use the same normalization parameters as in the training phase
        mfcc_data = (mfcc_data - mfcc_data.mean()) / mfcc_data.std()
        desired_mfcc_shape = (1, 862, 13)
        if mfcc_data.shape[1] < desired_mfcc_shape[1]:
            padding_width = desired_mfcc_shape[1] - mfcc_data.shape[1]
            mfcc_data = np.pad(mfcc_data, ((0, 0), (0, padding_width), (0, 0)), mode='constant')
        # Normalize ZCR value (if needed)
        zcr_value = 0.0
        with open(zcr_file_path, 'r') as file:
            # Read the first line and extract the numeric value
            line = file.readline()
            zcr_value = float(line.split(':')[1].strip())
        zcr_value = np.array([zcr_value])
        zcr_value = np.expand_dims(zcr_value, axis=0)
        
        # print(mfcc_data.shape)
        # Predict the class using the loaded model
        predictions = model.predict([spectrogram_data, mfcc_data, zcr_value])
        predicted_label = np.argmax(predictions, axis=1)

        # Assuming label_encoder_classes is a dictionary mapping class indices to class labels
        predicted_class_label = label_encoder_classes[predicted_label[0]]
        
        class_names = [
            'class_airport', 'class_bus', 'class_metro', 'class_metro_station', 'class_park',
            'class_public_square', 'class_shopping_mall', 'class_street_pedestrian', 
            'class_street_traffic', 'class_tram'
        ]

        # Print the file name and predicted class label
        print(f"File: {filename}, Predicted Class Label: {class_names[predicted_class_label]}")
        print("-----------------------------")

