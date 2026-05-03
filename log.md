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

---

## System B Train Results:

```
Loading features from features_systemB.npz ...
Feature matrix shape : (3000, 2048)
Labels shape         : (3000,)
Class balance        : happy(0)=1500, angry(1)=1500

Training set: 2400 samples
Testing  set: 600 samples

Feature scaling complete (StandardScaler fit on train only).

Running GridSearchCV (5-fold CV) — linear-only fine sweep ...
Fitting 5 folds for each of 11 candidates, totalling 55 fits

Grid search finished in 56.4 seconds.
Best parameters     : {'C': 0.005}
Best CV accuracy    : 0.8108

All hyperparameter combinations (sorted by CV accuracy):
  CV=0.8108 (+/- 0.0130)  {'C': 0.005}
  CV=0.8017 (+/- 0.0052)  {'C': 0.01}
  CV=0.7987 (+/- 0.0180)  {'C': 0.001}
  CV=0.7842 (+/- 0.0177)  {'C': 0.0005}
  CV=0.7804 (+/- 0.0092)  {'C': 5}
  CV=0.7804 (+/- 0.0092)  {'C': 0.1}
  CV=0.7804 (+/- 0.0092)  {'C': 0.5}
  CV=0.7804 (+/- 0.0092)  {'C': 10}
  CV=0.7804 (+/- 0.0092)  {'C': 1}
  CV=0.7796 (+/- 0.0097)  {'C': 0.05}
  CV=0.7646 (+/- 0.0105)  {'C': 0.0001}

=======================================================
           SYSTEM B - FINAL RESULTS
=======================================================
Training Accuracy : 94.79%
Testing  Accuracy : 82.50%
Train-Test Gap    : 12.29%
Fitting Quality   : Overfitting (large gap between train and test accuracy)
=======================================================

Classification Report (Test Set):
              precision    recall  f1-score   support

       happy     0.8328    0.8133    0.8229       300
       angry     0.8176    0.8367    0.8270       300

    accuracy                         0.8250       600
   macro avg     0.8252    0.8250    0.8250       600
weighted avg     0.8252    0.8250    0.8250       600

Confusion Matrix (Test Set):
                 Predicted
                happy  angry
Actual happy      244     56
Actual angry       49    251
```
