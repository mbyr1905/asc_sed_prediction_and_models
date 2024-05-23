from tensorflow.lite.python.interpreter import Interpreter

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path=r"D:\prediction\saved_models_tflite\converted_model.tflite")
interpreter.allocate_tensors()

# Print input details
print("Input details:")
print(interpreter.get_input_details())

# Print output details
print("\nOutput details:")
print(interpreter.get_output_details())