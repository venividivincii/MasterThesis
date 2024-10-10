import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Get CUDA device information
import pycuda.driver as cuda
import pycuda.autoinit
device = cuda.Device(0)
print("Device Name:", device.name())
print("Total Memory:", device.total_memory() / (1024 ** 2), "MB")
print("Compute Capability:", device.compute_capability())

import argparse
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from processtransformer import constants
from processtransformer.data.processor import LogsDataProcessor
from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer
import warnings
warnings.filterwarnings("ignore")

dataset = "helpdesk"
filename = "helpdesk.csv"
task = constants.Task.TIMES
dirpath = "./datasets"
datetime_format = "%Y-%m-%d %H:%M:%S.%f"
# datetime_format = "ISO8601"

# Process raw logs
start = time.time()
data_processor = LogsDataProcessor(dataset, 
    filepath=f"./datasets/{dataset}/{filename}",
    columns = ["Case ID", "Activity", "Complete Timestamp"], #["case:concept:name", "concept:name", "time:timestamp"], 
    dir_path=dirpath, pool = 1, datetime_format=datetime_format) #changed from 4 to 1
data_processor.process_logs(task=task, sort_temporally= True)
end = time.time()
print(f"Total processing time: {end - start}")

model_dir = "./models"
result_dir = "./results"
next_act_dir = "./results/next_activity"
next_time_dir = "./results/next_time"
remaining_time_dir = "./results/remaining_time"

epochs = 100
batch_size = 64
learning_rate = 0.002

# Create and save the model
model_path = f"{model_dir}/{dataset}"
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_path = f"{model_path}/predict_ckpt"

result_path = f"{result_dir}/{dataset}"
if not os.path.exists(result_path):
    os.makedirs(result_path)
result_path = f"{result_path}/results"

next_act_path = f"{next_act_dir}"
if not os.path.exists(next_act_path):
    os.makedirs(next_act_path)
next_act_path = f"{next_act_path}/next_act_predict"

next_time_path = f"{next_time_dir}"
if not os.path.exists(next_time_path):
    os.makedirs(next_time_path)
next_time_path = f"{next_time_path}/next_time_predict"

remaining_time_path = f"{remaining_time_dir}"
if not os.path.exists(remaining_time_path):
    os.makedirs(remaining_time_path)
remaining_time_path = f"{remaining_time_path}/remaining_time_predict"

# Create and load the data object
data_loader = loader.LogsDataLoader(name=dataset)
# Load next event data
(train_act_df, test_act_df, x_word_dict, y_word_dict, max_case_length,
 vocab_size, num_output) = data_loader.load_data(constants.Task.NEXT_ACTIVITY)
# Load time task data
(train_time_df, test_time_df, x_word_dict1, y_word_dict1, max_case_length1,
 vocab_size1, num_output1) = data_loader.load_data(constants.Task.TIMES)

# Prepare time task data
(train_token_x, train_time_x, 
 train_token_next_time, train_token_rmain, time_scaler, y_scaler, y_scaler1) = data_loader.prepare_data_times(
    train_time_df, x_word_dict1, max_case_length1)
# Prepare next event task data
train_token_act_x, train_token_act_y = data_loader.prepare_data_next_activity(
    train_act_df, x_word_dict, y_word_dict, max_case_length)
# Create the model
transformer_model = transformer.get_predict_model(
    max_case_length=max_case_length, 
    vocab_size=vocab_size, output_dim=num_output)
# Compile the model
transformer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss={
        'out1': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "out2": tf.keras.losses.LogCosh(),
        'out3': tf.keras.losses.LogCosh()
    },
    loss_weights={'out1': 0.6, "out2": 2, "out3": 0.3}
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_weights_only=True,
    monitor="loss", save_best_only=True)
# Train the model
transformer_model.fit(
    [train_token_act_x, train_token_x, train_time_x], 
    [train_token_act_y, train_token_next_time, train_token_rmain],
    epochs=epochs, 
    batch_size=batch_size,
    verbose=2, 
    callbacks=[model_checkpoint_callback]
)


######################################### Modified Evaluation Function #########################################

# Initialize dictionaries to store metrics and counts for each task
metrics_activity = {'k': [], 'weight': [], 'accuracy': [], 'fscore': [], 'precision': [], 'recall': []}
metrics_next_time = {'k': [], 'weight': [], 'MAE': [], 'MSE': [], 'RMSE': []}
metrics_remaining_time = {'k': [], 'weight': [], 'MAE': [], 'MSE': [], 'RMSE': []}
counts = []  # To store counts for each k

# Total number of test instances (to calculate weights later)
total_test_instances = len(test_act_df)

# Iterate over all possible prefix lengths
for i in range(max_case_length):
    test_data_subset = test_act_df[test_act_df["k"] == i]
    test_data_subset1 = test_time_df[test_time_df["k"] == i]
    
    # Only proceed if there are test instances for the current k
    if len(test_data_subset) > 0:
        # Prepare data for next activity prediction
        test_token_act_x, test_y1 = data_loader.prepare_data_next_activity(
            test_data_subset, x_word_dict, y_word_dict, max_case_length)
        
        # Prepare data for time-related predictions
        test_token_x, test_time_x, test_y2, test_y3, _, _, _ = data_loader.prepare_data_times(
            test_data_subset1, x_word_dict, max_case_length, time_scaler, y_scaler, y_scaler1, False)

        # Make predictions using the transformer model
        y_pred = transformer_model.predict([test_token_act_x, test_token_x, test_time_x])
        
        # -------------------- Next Activity Prediction Evaluation --------------------
        y_pred1 = np.argmax(y_pred[0], axis=1)  # Get class labels
        accuracy = metrics.accuracy_score(test_y1, y_pred1)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            test_y1, y_pred1, average="weighted")
        
        # Append metrics for next activity prediction
        metrics_activity['k'].append(i)
        metrics_activity['accuracy'].append(accuracy)
        metrics_activity['fscore'].append(fscore)
        metrics_activity['precision'].append(precision)
        metrics_activity['recall'].append(recall)
        
        # -------------------- Next Event Time Prediction Evaluation --------------------
        y_pred2 = y_pred[1]
        _test_y2 = y_scaler.inverse_transform(test_y2)
        _y_pred2 = y_scaler.inverse_transform(y_pred2)
        
        mae_next_time = metrics.mean_absolute_error(_test_y2, _y_pred2)
        mse_next_time = metrics.mean_squared_error(_test_y2, _y_pred2)
        rmse_next_time = np.sqrt(mse_next_time)
        
        # Append metrics for next event time prediction
        metrics_next_time['k'].append(i)
        metrics_next_time['MAE'].append(mae_next_time)
        metrics_next_time['MSE'].append(mse_next_time)
        metrics_next_time['RMSE'].append(rmse_next_time)
        
        # -------------------- Remaining Time Prediction Evaluation --------------------
        y_pred3 = y_pred[2]
        _test_y3 = y_scaler1.inverse_transform(test_y3)
        _y_pred3 = y_scaler1.inverse_transform(y_pred3)
        
        mae_remaining_time = metrics.mean_absolute_error(_test_y3, _y_pred3)
        mse_remaining_time = metrics.mean_squared_error(_test_y3, _y_pred3)
        rmse_remaining_time = np.sqrt(mse_remaining_time)
        
        # Append metrics for remaining time prediction
        metrics_remaining_time['k'].append(i)
        metrics_remaining_time['MAE'].append(mae_remaining_time)
        metrics_remaining_time['MSE'].append(mse_remaining_time)
        metrics_remaining_time['RMSE'].append(rmse_remaining_time)
        
        # Record the count for current k
        counts.append(len(test_data_subset))

# Calculate weights based on counts
counts = np.array(counts)
weights = counts / total_test_instances

# Add weights to each metrics dictionary
metrics_activity['weight'] = weights
metrics_next_time['weight'] = weights
metrics_remaining_time['weight'] = weights

# -------------------- Create DataFrames and Save to CSV --------------------

# Next Activity Prediction Results
activity_df = pd.DataFrame(metrics_activity)

# Calculate Weighted Mean for Next Activity Prediction
weighted_accuracy = np.average(activity_df['accuracy'], weights=activity_df['weight'])
weighted_fscore = np.average(activity_df['fscore'], weights=activity_df['weight'])
weighted_precision = np.average(activity_df['precision'], weights=activity_df['weight'])
weighted_recall = np.average(activity_df['recall'], weights=activity_df['weight'])

# Append Weighted Mean to the DataFrame using pd.concat
weighted_activity_df = pd.DataFrame({
    'k': ['Weighted Mean'],
    'weight': [''],
    'accuracy': [weighted_accuracy],
    'fscore': [weighted_fscore],
    'precision': [weighted_precision],
    'recall': [weighted_recall]
})

activity_df = pd.concat([activity_df, weighted_activity_df], ignore_index=True)

# Save Next Activity Prediction Results to CSV
activity_df.to_csv(result_path + "_activity.csv", index=False)

# Next Event Time Prediction Results
next_time_df = pd.DataFrame(metrics_next_time)

# Calculate Weighted Mean for Next Event Time Prediction
weighted_mae_next = np.average(next_time_df['MAE'], weights=next_time_df['weight'])
weighted_mse_next = np.average(next_time_df['MSE'], weights=next_time_df['weight'])
weighted_rmse_next = np.average(next_time_df['RMSE'], weights=next_time_df['weight'])

# Append Weighted Mean to the DataFrame using pd.concat
weighted_next_time_df = pd.DataFrame({
    'k': ['Weighted Mean'],
    'weight': [''],
    'MAE': [weighted_mae_next],
    'MSE': [weighted_mse_next],
    'RMSE': [weighted_rmse_next]
})

next_time_df = pd.concat([next_time_df, weighted_next_time_df], ignore_index=True)

# Save Next Event Time Prediction Results to CSV
next_time_df.to_csv(result_path + "_next_time.csv", index=False)

# Remaining Time Prediction Results
remaining_time_df = pd.DataFrame(metrics_remaining_time)

# Calculate Weighted Mean for Remaining Time Prediction
weighted_mae_remaining = np.average(remaining_time_df['MAE'], weights=remaining_time_df['weight'])
weighted_mse_remaining = np.average(remaining_time_df['MSE'], weights=remaining_time_df['weight'])
weighted_rmse_remaining = np.average(remaining_time_df['RMSE'], weights=remaining_time_df['weight'])

# Append Weighted Mean to the DataFrame using pd.concat
weighted_remaining_time_df = pd.DataFrame({
    'k': ['Weighted Mean'],
    'weight': [''],
    'MAE': [weighted_mae_remaining],
    'MSE': [weighted_mse_remaining],
    'RMSE': [weighted_rmse_remaining]
})

remaining_time_df = pd.concat([remaining_time_df, weighted_remaining_time_df], ignore_index=True)

# Save Remaining Time Prediction Results to CSV
remaining_time_df.to_csv(result_path + "_remaining_time.csv", index=False)

# -------------------- Optional: Print DataFrames --------------------
print("Next Activity Prediction Results:")
print(activity_df)

print("\nNext Event Time Prediction Results:")
print(next_time_df)

print("\nRemaining Time Prediction Results:")
print(remaining_time_df)