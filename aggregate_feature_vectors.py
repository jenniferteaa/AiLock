#The following code is used to produce .h5 files for the activations produced in the previous step.
# 1 .h5 file is produced for an input of all the activation.txt files under each part.


import os
import numpy as np
from os import listdir
from os.path import isfile, join
import h5py

# Path to the parent directory containing corresponding activations for images (one file per image)
image_dir = "/content/drive/MyDrive/google_split/activations_google_split"
out_file_name = "google_files"
nexus_flag = False  # True if the images are from the Google dataset
toys_flag = False    # True if the images are from the Aloi dataset

# Iterate through each part segment
for part_num in range(1, 6):
    part_dir = os.path.join(image_dir, f"part_{part_num}")
    if not os.path.exists(part_dir):
        continue

    print(f"Processing part {part_num}")

    # Aggregate feature vectors for the current part segment
    feature_vectors = []
    for subdir, _, files in os.walk(part_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                data = np.loadtxt(file_path, delimiter=',')
                feature_vectors.append(data)

    feature_vectors = np.array(feature_vectors)

    # Save aggregated feature vectors to .h5 file
    out_file = os.path.join(image_dir, f"{out_file_name}_part_{part_num}.h5")
    with h5py.File(out_file, 'w') as hf:
        hf.create_dataset('dataset', data=feature_vectors)

    print(f"Saved aggregated feature vectors to {out_file}")

print("Finished processing all part segments.")

#output messages
#Processing part 1
#Saved aggregated feature vectors to /content/drive/MyDrive/google_split/activations_google_split/google_files_part_1.h5
#Processing part 2
#Saved aggregated feature vectors to /content/drive/MyDrive/google_split/activations_google_split/google_files_part_2.h5
#Processing part 3
#Saved aggregated feature vectors to /content/drive/MyDrive/google_split/activations_google_split/google_files_part_3.h5
#Processing part 4
#Saved aggregated feature vectors to /content/drive/MyDrive/google_split/activations_google_split/google_files_part_4.h5
#Processing part 5
#Saved aggregated feature vectors to /content/drive/MyDrive/google_split/activations_google_split/google_files_part_5.h5
#Finished processing all part segments.