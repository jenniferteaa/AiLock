#This step is used to produce activation files for each image in a given dataset.
#We implement the Inceptionv3 model's mixed10 layer for better feature extraction.
#This code produces 5 parts for each of the 3 datasets, each part would have activation files for its respective image.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

def compute_activations(input_path, output_path, desired_layer, bottleneck_tensor_name, bottleneck_tensor_size):
    # Load the InceptionV3 model with include_top=False to exclude the fully connected layers
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling=None)

    # Get the desired layer output
    desired_layer_output = base_model.get_layer(desired_layer).output

    # Define a new model with the desired layer as output
    model = tf.keras.Model(inputs=base_model.input, outputs=desired_layer_output)

    # Process each part directory
    for part_dir in os.listdir(input_path):
        part_path = os.path.join(input_path, part_dir)
        if os.path.isdir(part_path):
            part_output_path = os.path.join(output_path, part_dir)
            os.makedirs(part_output_path, exist_ok=True)
            activations_output_path = os.path.join(part_output_path, 'activations')
            os.makedirs(activations_output_path, exist_ok=True)

            # Process each image in the part directory
            for filename in os.listdir(part_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(part_path, filename)
                    try:
                        img = image.load_img(img_path, target_size=(299, 299))
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        continue

                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)

                    # Get activations for the image
                    activations = model.predict(x)

                    # Save activations to a text file
                    output_filename = os.path.splitext(filename)[0] + '.txt'
                    output_filepath = os.path.join(activations_output_path, output_filename)
                    np.savetxt(output_filepath, activations.flatten(), delimiter=',')

# Set input and output paths
input_path = '/content/drive/MyDrive/google_split/Cropped_5overlapping_sqr'  # Update with your input path
output_path = '/content/drive/MyDrive/google_split/activations_google_split'  # Update with your output path

# Define desired layer and corresponding bottleneck tensor
desired_layer = 'mixed10'  # Layer name for mixed10:0
bottleneck_tensor_name = 'mixed10'
bottleneck_tensor_size = 2048

# Compute activations for each part
compute_activations(input_path, output_path, desired_layer, bottleneck_tensor_name, bottleneck_tensor_size)
