# KalmanNet_fertilizereva
KNet/
This folder is used to store the RNN model and trained model

KNet_CNN/
This folder is used to store the CNN model and trained model
contain both Kalman Filter to init the sliding window and ground truth to init the sliding window

KNet_UNet/
This folder is used to store the UNet model(3 or 4 layers) and trained model

Pipeline/
These are the pipeline files for RNN, CNN(UNet) cases of KalmanNet respectively. 
The pipeline mainly defines the Training/CV/Testing processes of KalmanNet.

RNN model -> main_for_fertilizer400.py
CNN model -> main_for_CNN.py
UNet model -> main_for_UNet.py
