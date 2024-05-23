import os
import tensorflow as tf

# Load your trained tf model
model = tf.keras.models.load_model(r"D:\prediction\saved_models_tf\audio_classification.hdf5")

# Convert the model to tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Specify the directory where you want to save the tflite model
output_dir = r"D:\prediction\saved_models_tflite"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the tflite model to the specified directory
output_path = os.path.join(output_dir, 'converted_model_sed.tflite')
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model converted and saved to: {output_path}")
