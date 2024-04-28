# AiLock
This project implements AiLock, an alternative to biometrics proposed and published in the research paper 'A Secure Mobile Authentication Alternative to Biometrics.

ai.lock is introduced as a secret image based authentication method for mobile devices which uses an imaging sensor to reliably extract authentication credentials similar to biometrics. In this project, we Implement ai.lock and test its efficiency particularly its Error Tolerance Threshold (ETT).

## Error Tolerance Threshold
ETT in terms of ai.lock represents the threshold used to separate valid from invalid authentication samples. Lower τ values indicate a stricter threshold for accepting matches between images. Given a reference image (the user initially sets as the password) and a candidate image (the image that the user has clicked in order to pass the authentication) Error Tolerance Threshold serves as a critical parameter in determining the level of similarity required between reference and candidate images for them to be considered a match. Adjusting this threshold allows system designers to balance between accommodating variations and maintaining precision in image recognition tasks.

Higher values of τ (e.g., τ = 7.80) indicate a higher tolerance for errors or discrepancies between images. In other words, the system is more lenient in accepting matches.
Lower values of τ (e.g., τ = 6.82) suggest a stricter threshold for accepting matches, meaning the system is more selective and requires a higher degree of similarity between images to accept them as matches.

## Implementing and Running AiLock

### Preparing the Dataset

We utilized three datasets for this project, all containing images of toy objects:

- Nexus dataset
- Google dataset
- Toys dataset
In our implementation, all datasets mentioned above contain object pictures, specifically toys. However, the authors in the paper have included various objects, such as scenery, bracelets, cloth designs, earrings, etc., in their dataset.

Step 1: Download the mentioned datasets.

Step 2: Image Splitting

We used the split_images.py code to split each image into multiple segments. Each image from the three datasets is split or cropped into five overlapping segments. This is done to ensure that each feature in the image is considered in multiple segments, reducing the impact of variations in pose, lighting, or expression. The splitting is also beneficial for enhanced robustness, improved feature representation, and reduction of dimensionality.

### Creation of Activation Files for Each Image

Run the activations.py code to create activations for each image. For this, we implemented a DNN with the InceptionV3's mixed10 layer for efficient feature extraction.

The file directory structure should look like this:

```
Nexus
└── Nexus_split
    ├── Cropped_images
    │   ├── part1
    │   ├── part2
    │   ├── part3
    │   ├── part4
    │   └── part5
    └── Nexus_Activations
        ├── nexus_part1.txt
        ├── nexus_part2.txt
        ├── nexus_part3.txt
        ├── nexus_part4.txt
        └── nexus_part5.txt

Google
└── Google_split
    ├── Cropped_images
    │   ├── part1
    │   ├── part2
    │   ├── part3
    │   ├── part4
    │   └── part5
    └── Google_Activations
        ├── google_part1.txt
        ├── google_part2.txt
        ├── google_part3.txt
        ├── google_part4.txt
        └── google_part5.txt

Toys
└── Toys_split
    ├── Cropped_images
    │   ├── part1
    │   ├── part2
    │   ├── part3
    │   ├── part4
    │   └── part5
    └── Toys_Activations
        ├── toys_part1.txt
        ├── toys_part2.txt
        ├── toys_part3.txt
        ├── toys_part4.txt
        └── toys_part5.txt
```

If the part1 folder under the Nexus dataset contains 'n' images, the nexus_part1 under the Nexus_Activations would have 'n' number of .txt files. These are the activation files.

### Aggregating the Activation Files

Run the code aggregate_feature_vectors.py to produce one .h5 file by aggregating all the activation files under a certain part. This file will then be used to train and evaluate ai.lock performance.

The resulting file generation messages should look something like this:

#output messages
#Processing part 1
#Saved aggregated feature vectors to /content/drive/MyDrive/google_split/activations_google_split/google_files_part_1.h5
...
#Finished processing all part segments.


### Organize the .h5 Files

Put the .h5 files corresponding to the full-size images under the full_size directory, and the .h5 files corresponding to each of the five image segments under the part_1 to part_5 directories. The final directory structure for the datasets should look like the following:

```
Datasets
└── Mixed10
    ├── full_size
    │   ├── nexus.h5
    │   ├── google.h5
    │   └── toys.h5
    ├── part_1
    │   ├── nexus_part1.h5
    │   ├── google_part1.h5
    │   └── toys_part1.h5
    ├── part_2
    │   ├── nexus_part2.h5
    │   ├── google_part2.h5
    │   └── toys_part2.h5
    ├── part_3
    │   ├── nexus_part3.h5
    │   ├── google_part3.h5
    │   └── toys_part3.h5
    ├── part_4
    │   ├── nexus_part4.h5
    │   ├── google_part4.h5
    │   └── toys_part4.h5
    └── part_5
        ├── nexus_part5.h5
        ├── google_part5.h5
        └── toys_part5.h5

```

### Dataset Splitting

For the Inception layer, run split_datasets_to_test_train.py to split the embeddings in each dataset into test (holdout) and train sets.

### Package Installation and Configuration

Download the Nearpy package.
Replace the last line of the hash_vector method in hashes/permutations/randombinaryprojections.py with the provided code snippet. (i.e. return [''.join(['1' if x > 0.0 else '0' for x in projection])]) to the following code:


```
return ((projection > 0).astype(int), (projection <= 0).astype(int))
```

### Running Experiments

To conduct single-layer single-image experiments:

Use the code provided under the Single_image directory.
For computing the best performing thresholds for binary classifications of images, use single_image_cv_train.py.
To evaluate the performance of ai.lock on the holdout set, use single_image_cv_holdout.py.
