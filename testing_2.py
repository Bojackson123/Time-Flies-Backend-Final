import os, cv2, csv, pickle, json
from analyze_video import pretrain_regression, analyze_video, calculate_accumulators
from joblib import dump, load


"""
    VIDEO PROCESSING: SPLITTING EACH VIDEO INTO FRAMES
"""
def convert_video_to_frames(video_path, save_dir):
    # Make sure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Determine video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        # Read a frame
        success, frame = cap.read()

        # If the frame was not successfully read, break the loop
        if not success:
            break

        # Save every frame without trying to simulate 30 FPS
        save_path = os.path.join(save_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(save_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'All frames have been saved to {save_dir}. Total frames: {frame_count}.')
    return original_fps

# # Define the directories
# source_dir = 'test_videos'
# target_dir = 'video_frames'


# videos = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]

# # Process each video
# for video in videos:
#     video_path = os.path.join(source_dir, video)
#     video_name = os.path.splitext(video)[0]
#     save_dir = os.path.join(target_dir, f'{video_name}_frames')
    
#     # Convert video and get the original fps
#     fps = convert_video_to_frames(video_path, save_dir)
#     print(f'Processed {video}: original FPS was {fps}.')
    
"""
    VIDEO PROCESSING: SPLITTING EACH VIDEO INTO FRAMES
"""

###########################################################################################

"""
    CREATE A LIST OF FRAME PATHS FOR EACH TRIAL: (START, END)
    FORMAT:
    [[frame_path_1, frame_path_2, ..., frame_path_n], [frame_path_1, frame_path_2, ..., frame_path_n], ...]
    
    ***RETURNS: ALL_FRAME_PATHS_LIST***
"""
# # Reads the Video_trial_sequence 
def read_video_csv(file_path):
    video_data = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    trial_ID = int(row['\ufeffTrial '])
                    video_id = int(row[' Video '])
                    start_frame = int(row[' Start_frame'])
                    end_frame = int(row[' End_frame '])
                    video_data[trial_ID] = (video_id, start_frame, end_frame)
                except KeyError as e:
                    print(f"KeyError encountered: {e}")
                    print(f"Available keys in the row: {list(row.keys())}")
                    break  # Break to only print the error for the first row
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return video_data

# # Defines path to the Video_trial_sequence.csv
# video_csv_path = 'test_csv_data/video_9.csv'

# # Creates a dictionary of the video data in this format:
# # {trial_id: (video_id, start_frame, end_frame)}
# video_trial_data = read_video_csv(video_csv_path)
# #print(video_trial_data)

# # creates a list of frame paths for each trial: (start, end)
# def create_frame_path_list(video_id, start_frame, end_frame):
#     frame_paths = []
#     for frame_num in range(start_frame, end_frame+1):
#         frame_path = f'video_frames/video_{video_id}_frames/frame_{frame_num:04d}.jpg'
#         frame_paths.append(frame_path)
#     return frame_paths

# # This function creates a list of lists, where each inner list contains the frame paths for a specific trial
# def generate_all_frame_paths(video_trial_data):
#     all_frame_paths = []
#     for trial_id, (video_id, start_frame, end_frame) in video_trial_data.items():
#         frame_list = create_frame_path_list(video_id, start_frame, end_frame)
#         all_frame_paths.append(frame_list)
#     return all_frame_paths

# def save_with_pickle(data, file_path):
#     with open(file_path, 'wb') as file:
#         pickle.dump(data, file)
        
# def load_with_pickle(file_path):
#     with open(file_path, 'rb') as file:
#         data = pickle.load(file)
#     return data

# # Use the function to generate the list of frame paths for all trials
# list_of_frame_paths = generate_all_frame_paths(video_trial_data)

# # Create a pickle file to store the list of frame paths for later use
# pickle_file_path = 'test_csv_data/list_of_frame_paths.pkl'
# save_with_pickle(list_of_frame_paths, pickle_file_path)

# ALL_FRAME_PATHS_LIST = load_with_pickle(pickle_file_path)

"""
    CREATE A LIST OF FRAME PATHS FOR EACH TRIAL: (START, END)
    FORMAT:
    [[frame_path_1, frame_path_2, ..., frame_path_n], [frame_path_1, frame_path_2, ..., frame_path_n], ...]
    
    ***RETURNS: ALL_FRAME_PATHS_LIST***
"""

############################################################################################

"""
    PRETRAIN THE REGRESSION MODEL WITH ALL_FRAME_PATHS_LIST AND SAVE IT FOR LATER ESTIMATIONS
    
    ***RETURNS: REGRESSION_MODEL***
"""
# # read the baseline_csv file and pull its still estimates
# def read_machine_of_still_tuned_csv(file_path):
#     baseline_data = {}
#     try:
#         with open(file_path, mode='r', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 try:
#                     trial_ID = int(row['trial id'])
#                     duration = float(row['\ufeffreal durations'])
#                     estimation = float(row['estimations'])                     
#                     acc_string = row['accumulator_floats_string']
#                     acc = [float(x.strip()) for x in acc_string.split()]
#                     baseline_data[trial_ID] = (duration, acc, estimation)
#                 except KeyError as e:
#                     print(f"KeyError encountered: {e}")
#                     print(f"Available keys in the row: {list(row.keys())}")
#                     break  # Break to only print the error for the first row
#                 except ValueError as e:
#                     print(f"ValueError encountered: {e}")
#                     print(f"Problematic row: {row}")
#                     break  # Break to only print the error for the first row
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#     return baseline_data

# file_path = 'test_csv_data/still_tuned_cleaned_data.csv'

# still_tuned = read_machine_of_still_tuned_csv(file_path)
# x_values = []
# y_values = []
# for trialID, (duration, acc, estimation) in still_tuned.items():
#     x_values.append(acc)
#     y_values.append(duration)
# # print(x_values)
# # print(y_values)

# # # Run the pretrain_regression function to train the regression model
# REGRESSION_MODEL = pretrain_regression(x_values, y_values)

# # # Save the regression model to a file (only done once and save for later use)
# dump(REGRESSION_MODEL, 'pretrained_regression/regression_model.joblib')

# # # On subsequent runs, load the regression model from the file instead of waiting to train again
REGRESSION_MODEL = load('pretrained_regression/regression_model.joblib')

"""
    PRETRAIN THE REGRESSION MODEL WITH ALL_FRAME_PATHS_LIST AND SAVE IT FOR LATER ESTIMATIONS
    
    ***RETURNS: REGRESSION_MODEL***
"""

####################################################################################################################

"""
    PREDICT THE ESTIMATES FOR EACH TRIAL USING THE REGRESSION MODEL
"""

def get_frame_paths(save_dir, start_frame, end_frame):
    frame_files = sorted(os.listdir(save_dir))  # Sort to maintain order
    # Ensure the selected indices are within the available range of frames
    if end_frame >= len(frame_files):
        end_frame = len(frame_files) - 1
    # Select the subset of frame paths based on start_frame and end_frame
    subset_frame_paths = [os.path.join(save_dir, frame_files[i]) for i in range(start_frame, end_frame + 1)]
    return subset_frame_paths

# read the baseline_csv file and pull its still estimates
def read_baseline_csv(file_path):
    baseline_data = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    trial_ID = int(row['trial_id'])
                    human_est = float(row[' human_estimates'])
                    expected_est = float(row[' still_estimates'])
                    duration = float(row[' real_duration'])
                    baseline_data[trial_ID] = (expected_est, human_est, duration)
                except KeyError as e:
                    print(f"KeyError encountered: {e}")
                    print(f"Available keys in the row: {list(row.keys())}")
                    break  # Break to only print the error for the first row
                except ValueError as e:
                    print(f"ValueError encountered: {e}")
                    print(f"Problematic row: {row}")
                    break  # Break to only print the error for the first row
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return baseline_data


video_data = read_video_csv('test_csv_data/video_9.csv')
# tuned_data = read_machine_of_still_tuned_csv('test_csv_data/still_tuned_cleaned_data_smaller.csv')
baseline_data = read_baseline_csv('test_csv_data/summary_baseline.csv')


if video_data is None is None or baseline_data is None:
    print("One or more CSV files could not be read properly.")


subsets = {
    trial_ID: (
        video_data[trial_ID][1], 
        video_data[trial_ID][2], 
        baseline_data[trial_ID][0], 
        baseline_data[trial_ID][1],
        baseline_data[trial_ID][2]
    ) 
    for trial_ID in video_data 
    if trial_ID in baseline_data and trial_ID in video_data
}

# subsets = {
#     TrialID: (VideoID, Start, End, Still_Estimate, Human_Estimate, real_duration),
# }

# Prepare a dictionary to hold all comparisons
estimate_comparisons = {}

# # Prep a list to hold accumulators for ever trial and list for associated durations.
# acc_combined = []
# time_combined = []

# Loop through all subsets pull the frame paths and predict the estimates
for trial_ID, (start_frame, end_frame, expected_estimate, human_estimate, real_duration) in subsets.items():
    
    # Generate frame paths for the current subset
    frame_dir = f'video_frames/video_{video_data[trial_ID][0]}_frames/'
    subset_frame_paths = get_frame_paths(frame_dir, start_frame, end_frame)
    
    # acc, time = calculate_accumulators(subset_frame_paths)
    # acc_combined.append(acc)
    # time_combined.append(time)
    # print(f"Trial {trial_ID}: Accumulator: {acc}, Time: {time}")
    # print(acc_combined)
    # print(time_combined)
    
    # create a directory to store the results
    results_dir = f'test_results/trials/trial_{trial_ID}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Define paths for the figure and JSON output
    figure_name = f'figure_{trial_ID}.png'
    figure_path = os.path.join(results_dir, figure_name)
    json_name = f'json_{trial_ID}.json'
    json_path = os.path.join(results_dir, json_name)

    # Analyze the video 
    figure, jsonData = analyze_video(subset_frame_paths, REGRESSION_MODEL)

# with open('training_pickle/training_x.pkl', 'wb') as f:
#     pickle.dump(acc_combined, f)
# with open('training_pickle/training_y.pkl', 'wb') as f:
#     pickle.dump(time_combined, f)    
    
    # Save the figure and JSON output
    figure.savefig(figure_path)
    with open(json_path, 'w') as f:
        f.write(jsonData)
        
    # Read the actual estimate from the .json data
    with open(json_path, 'r') as f:
        results = json.load(f)
        actual_estimate = results["charts"][-1]["data"]["est"]
        
    # Compare the expected and actual estimates
    comparison = {
        "Real Duration": real_duration,
        "Human Estimate": human_estimate,
        "Expected": expected_estimate,
        "Actual": actual_estimate,
    }
    estimate_comparisons[trial_ID] = comparison
    
    print(f"Trial {trial_ID} comparison: Expected {expected_estimate}, Actual {actual_estimate}")   
    print(f"Trial {trial_ID} analysis complete. Results saved.")
    
# Assuming all trials have been processed, save the comparisons
comparison_path = f'test_results/estimate_comparison.json'
with open(comparison_path, 'w') as f:
    json.dump(estimate_comparisons, f, indent=4)

print(f"Estimate comparisons saved to {comparison_path}.")
    
    
"""
    PREDICT THE ESTIMATES FOR EACH TRIAL USING THE REGRESSION MODEL
"""

