
import os
import cv2
import time, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from joblib import load

from src.parameters import Parameters
from src.AlexNetNew import AlexNet
from src.updateNetwork import UpdateNetwork
from src.calculate_accumulators import Accummulators

from src.updateNetworkData import ExportUpdateNetworkData

import warnings
warnings.filterwarnings("ignore")

def analyze_video(frames):
    """
    Analyzes a video by processing each frame and predicting the time using a trained regression model.

    Args:
        frames (str): The directory path containing the frames of the video.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the figures generated for each attention value, and the second dictionary contains the JSON data exported for each attention value.
    """
    # Load the frames from the given directory
    frame_paths = [os.path.join(frames, file) for file in os.listdir(frames) if file.endswith(".png")]

    # Load the trained regression model
    models_list = load('../pretrained_regression/regression_model.joblib')
    
    # Initialize the AlexNet model and create a dictionary to store the figures and JSON data
    alexnet = AlexNet()
    figures = {}
    json_datas = {}
    
    # Different attention % of baseline values
    c_values = [1/.5, 1/1, 1/1.5, 1/2]
    upNets = {}
    params_list = {}
    
    # Create an instance of Parameters and UpdateNetwork for each C value
    name_counter = 0.0
    for c in c_values:
        name_counter += 1.0
        param = Parameters(C=c)
        params_list[c] = param.params
        upNets[name_counter] = UpdateNetwork(params_list[c])

    for frame_path in frame_paths:
        # Load the frame and shape it
        frame = cv2.imread(frame_path)
        frame = shape_frame(frame)

        # Run the frame through the AlexNet to get the distances at each layer
        label = alexnet.run(frame)

        # Run the frame through each UpdateNetwork instance and calculate the accumulators
        for c, upNet in upNets.items():
            upNet.run(frame, alexnet.features, alexnet.output_prob, alexnet.states, alexnet.labels)

    # Pull the accumulator values from each UpdateNetwork instance and predict the time
    for c, upNet in upNets.items():
        # Convert the accumulator values to a 1x4 array
        acc = list(upNet.accumulator.values())
        acc_array = np.array(acc).reshape(1, -1)
        
        estimate = average_predict(models_list, acc_array)
        
        # Plots the graphs and exports the data to JSON
        figure = upNet.plot(alexnet.states, alexnet.labels, False, estimate)
        figures[c] = figure
        
        export = ExportUpdateNetworkData(upNet)
        json_datas[c] = export.to_json()
    
    # Return the figures and JSON data for each attention value as a dict
    return figures, json_datas

def average_predict(models, X_new):
    """
    Makes predictions with all models and averages the predictions.

    Args:
        models (list): A list of trained regression models.
        X_new (ndarray): The input data for prediction.

    Returns:
        ndarray: The averaged predictions.
    """
    # Make predictions with all models
    predictions = np.array([model.predict(X_new) for model in models])
    # Average the predictions across all models
    return np.mean(predictions, axis=0)    

def shape_frame(frame):
    """
    Shapes the frame to a square by cropping the longer side.

    Args:
        frame (ndarray): The input frame.

    Returns:
        ndarray: The shaped frame.
    """
    if np.shape(frame)[0] > np.shape(frame)[1]:
        onset = np.shape(frame)[0] - np.shape(frame)[1]
        return frame[int(onset/2):-int(onset/2), :]
    elif np.shape(frame)[0] < np.shape(frame)[1]:
        onset = np.shape(frame)[1] - np.shape(frame)[0]
        return frame[:, int(onset/2):-int(onset/2), :]
    return frame    


def calculate_accumulators(frames):
    """
    Calculates the accumulators for a set of frames.

    Args:
        frames (str): The directory path containing the frames.

    Returns:
        tuple: A tuple containing the accumulator values and the estimated time.
    """
    frame_path = frames
    
    params = Parameters().params
    params['visuals'] = True
    alexnet = AlexNet()
    upNet = UpdateNetwork(params)
    t = 0
    
    while len(frame_path) > 0:
        frame = cv2.imread(frame_path.pop(0))
        t += 1
        if np.shape(frame)[0] > np.shape(frame)[1]:
            onset = np.shape(frame)[0] - np.shape(frame)[1]
            frame = frame[int(onset/2):-int(onset/2), :]
        elif np.shape(frame)[0] < np.shape(frame)[1]:
            onset = np.shape(frame)[1] - np.shape(frame)[0]
            frame = frame[:, int(onset/2):-int(onset/2), :]

        label = alexnet.run(frame)
        upNet.run(frame, alexnet.features, alexnet.output_prob, alexnet.states, alexnet.labels)
    
    acc = list(upNet.accumulator.values())
    time = t / params['FPS']
    return acc, time

def pretrain_regression():
    """
    Pretrains a regression model using the training data.

    Returns:
        list: A list of trained regression models.
    """
    file_x_path = 'training_pickle/training_x.pkl'
    with open(file_x_path, 'rb') as file:
        x_train_list = pickle.load(file)
   
    file_y_path = 'training_pickle/training_y.pkl'
    with open(file_y_path, 'rb') as file:
        y_train_list = pickle.load(file)
        
    # Convert lists to NumPy arrays
    train_x = np.array(x_train_list)
    train_y = np.array(y_train_list)
    
    # # Define an ε-SVR model with specified parameters
    # model = SVR(kernel='rbf', C=1e3, gamma=1e-4)
    # model.fit(train_x, train_y)
    
    kf = KFold(n_splits=10)
    model_performance = []
    model_errors_mse = []
    model_errors_mae = []
    model_errors_mne = []  # List to store each fold's mean normalized error

    models = []
    for train_index, test_index in kf.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

        model = SVR(kernel='rbf', C=1e3, gamma=1e-4)
        model.fit(X_train, y_train)
        models.append(model)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mne = np.mean(np.abs(y_pred - y_test) / (np.abs(y_test) + 1e-10))  
        
        model_errors_mse.append(mse)
        model_errors_mae.append(mae)
        model_errors_mne.append(mne)

        score = model.score(X_test, y_test)
        model_performance.append(score)

    print("Average model performance (R^2):", np.mean(model_performance))
    print("Average model mean squared error:", np.mean(model_errors_mse))
    print("Average model mean absolute error:", np.mean(model_errors_mae))
    print("Average model mean normalized error:", np.mean(model_errors_mne))

    return models


# def pretrain_regression(all_data):
#     training_x = []
#     training_y = []
#     trial_count = 0
    
#     for filename, dists in all_data.items():
#         trial_count += 1
#         print(f"Trial {trial_count} from file {filename}")
        
#         # Instantiate Accummulators for each trial
#         ac = Accummulators()
#         params = Parameters().params
        
#         # Call calculate with the entire time series data for each layer
#         final_x, thresh_, extra_dp = ac.calculate(dists, extra_dp_freq=0)
        
#         # The final_x here should be the output from processing the entire time series
#         training_x.append(final_x)
#         training_y.append(len(dists[0]) / params['FPS'])
#         # print("Training X:", training_x)
#         # print("Training Y:", training_y[-1])
    
#     # Pickle training_x and training_y for later use
#     with open('training_pickle/training_x.pkl', 'wb') as f:
#         pickle.dump(training_x, f)
#     with open('training_y.pkl', 'wb') as f:
#         pickle.dump(training_y, f)
        
#     # Convert lists to NumPy arrays
#     train_x = np.array(training_x)
#     train_y = np.array(training_y)
    
#     # # Define an ε-SVR model with specified parameters
#     # model = SVR(kernel='rbf', C=0.001, gamma=0.0001)
#     # model.fit(train_x, train_y)
    
#     kf = KFold(n_splits=10)
#     model_performance = []
#     model_errors_mse = []
#     model_errors_mae = []
#     model_errors_mne = []  # List to store each fold's mean normalized error

#     for train_index, test_index in kf.split(train_x):
#         X_train, X_test = train_x[train_index], train_x[test_index]
#         y_train, y_test = train_y[train_index], train_y[test_index]

#         model = SVR(kernel='rbf', C=1e3, gamma=1e-4)
#         model.fit(X_train, y_train)
        
#         y_pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         mne = np.mean(np.abs(y_pred - y_test) / (np.abs(y_test) + 1e-10))  # Calculate mean normalized error, adding a small number to avoid division by zero
        
#         model_errors_mse.append(mse)
#         model_errors_mae.append(mae)
#         model_errors_mne.append(mne)

#         score = model.score(X_test, y_test)
#         model_performance.append(score)

#     print("Average model performance (R^2):", np.mean(model_performance))
#     print("Average model mean squared error:", np.mean(model_errors_mse))
#     print("Average model mean absolute error:", np.mean(model_errors_mae))
#     print("Average model mean normalized error:", np.mean(model_errors_mne))

#     return model
    
  

# # TESTING CODE
# def analyze_video(frames, models_list):
       
#     # frame_path = []
#     # for file in os.listdir(frames):
#     #     if file.endswith(".png"):
#     #         frame_path.append(os.path.join(frames, file))
    
#     frame_path = frames
    
     
#     params = Parameters().params
#     params['C'] = 1
#     params['visuals'] = True
#     alexnet = AlexNet()
#     upNet = UpdateNetwork(params)
#     ac = Accummulators()
#     estimate = 0
#     frames = frame_path.copy()
    
#     while len(frames) > 0:
            
#         frame = cv2.imread(frames.pop(0))

#         if np.shape(frame)[0] > np.shape(frame)[1]:
#             onset = np.shape(frame)[0] - np.shape(frame)[1]
#             frame = frame[int(onset/2):-int(onset/2), :]
#         elif np.shape(frame)[0] < np.shape(frame)[1]:
#             onset = np.shape(frame)[1] - np.shape(frame)[0]
#             frame = frame[:, int(onset/2):-int(onset/2), :]

#         label = alexnet.run(frame)
#         upNet.run(frame, alexnet.features, alexnet.output_prob, alexnet.states, alexnet.labels)
#         # distance = upNet.distances
#         # final_x, thresh_, extra_dp = ac.calculate(distance, extra_dp_freq=0)
    
#     acc = list(upNet.accumulator.values())
#     acc2 = np.array(acc).reshape(1, 4)
#     print("Accumulator:", acc)
#     print("Accumulator:", acc2)  
#     # Estimation
#     estimate = average_predict(models_list, acc2)


#     figure = upNet.plot(alexnet.states, alexnet.labels, False, estimate)

#     export = ExportUpdateNetworkData(upNet)
#     json_data = export.to_json()
    
#     return figure, json_data



