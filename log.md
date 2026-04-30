## Log files

Date: 4/30/2026

# System B
- added the requirements.txt.
- if the libraries are not previously installed, please create a venv and pip install the libraries from the requirements.txt

1. python -m venv env
2. env\Scripts\activate
3. pip install -r requirements.txt

### Result after preprocessing and Feature Extraction:

Feature matrix shape : (3000, 2048)
Labels shape         : (3000,)
Total images processed: 3000

**Note:**
    The features that are extracted from the ResNet50 model is stored into `features_systemB.npz` file as we don't have to 
    process all the 3000 images everytime.

    During the training and evaluation the features from the .npz files can be used.
