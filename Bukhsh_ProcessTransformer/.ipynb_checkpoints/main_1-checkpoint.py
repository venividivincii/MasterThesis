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

# Imports
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics 

from package.processtransformer import constants
from package.processtransformer.models import transformer
from package.processtransformer.data.loader import LogsDataLoader
from package.processtransformer.data.processor import LogsDataProcessor

data_dir = "./datasets/"
if not os.path.exists(data_dir): 
  os.mkdir(data_dir)

dataset_name = "bpi_2012"
data_processor = LogsDataProcessor(name=dataset_name, filepath="BPI_Challenge_2012.csv",  
                                    columns = ["case:concept:name", "concept:name", "time:timestamp"], #specify the columns name containing case_id, activity name and timestamp 
                                    dir_path='datasets', datetime_format="ISO8601", pool = 4)
data_processor.process_logs(task=constants.Task.NEXT_ACTIVITY, sort_temporally= True)

# Load data
data_loader = LogsDataLoader(name = dataset_name)

(train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
    vocab_size, num_output) = data_loader.load_data(constants.Task.NEXT_ACTIVITY)

# Prepare training examples for next activity prediction task
train_token_x, train_token_y = data_loader.prepare_data_next_activity(train_df, 
    x_word_dict, y_word_dict, max_case_length)

learning_rate = 0.001
batch_size = 12
epochs = 100

# Create and train a transformer model
transformer_model = transformer.get_next_activity_model(
    max_case_length=max_case_length, 
    vocab_size=vocab_size,
    output_dim=num_output)

transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

transformer_model.fit(train_token_x, train_token_y, 
    epochs=epochs, batch_size=batch_size)

# Initialize lists to store metrics and counts
k_list, accuracies, fscores, precisions, recalls = [], [], [], [], []
weighted_accuracies, weighted_fscores, weighted_precisions, weighted_recalls = [], [], [], []
num_instances_list = []
total_instances = 0

# Loop over each prefix length
for i in range(max_case_length):
    test_data_subset = test_df[test_df["k"] == i]
    num_instances = len(test_data_subset)
    k_list.append(i)
    num_instances_list.append(num_instances)
    
    if num_instances > 0:
        total_instances += num_instances
        test_token_x, test_token_y = data_loader.prepare_data_next_activity(
            test_data_subset, x_word_dict, y_word_dict, max_case_length
        )
        y_pred = np.argmax(transformer_model.predict(test_token_x), axis=1)
        
        accuracy = metrics.accuracy_score(test_token_y, y_pred)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            test_token_y, y_pred, average="weighted"
        )
        
        accuracies.append(accuracy)
        fscores.append(fscore)
        precisions.append(precision)
        recalls.append(recall)
        
        weighted_accuracies.append(accuracy * num_instances)
        weighted_fscores.append(fscore * num_instances)
        weighted_precisions.append(precision * num_instances)
        weighted_recalls.append(recall * num_instances)
    else:
        # If there are no instances for this prefix, append zeros
        accuracies.append(0)
        fscores.append(0)
        precisions.append(0)
        recalls.append(0)
        weighted_accuracies.append(0)
        weighted_fscores.append(0)
        weighted_precisions.append(0)
        weighted_recalls.append(0)

# Compute weights for each prefix
weights = [n / total_instances if total_instances > 0 else 0 for n in num_instances_list]

# Create a DataFrame with the collected metrics
df = pd.DataFrame({
    'k': k_list,
    'weight': weights,
    'accuracy': accuracies,
    'fscore': fscores,
    'precision': precisions,
    'recall': recalls
})

# Compute weighted average metrics
average_accuracy = sum(weighted_accuracies) / total_instances if total_instances > 0 else 0
average_fscore = sum(weighted_fscores) / total_instances if total_instances > 0 else 0
average_precision = sum(weighted_precisions) / total_instances if total_instances > 0 else 0
average_recall = sum(weighted_recalls) / total_instances if total_instances > 0 else 0

# Append the weighted averages to the DataFrame
weighted_mean_row = {
    'k': 'Weighted Mean',
    'weight': '',
    'accuracy': average_accuracy,
    'fscore': average_fscore,
    'precision': average_precision,
    'recall': average_recall
}
df = pd.concat([df, pd.DataFrame([weighted_mean_row])], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv(f"{dataset_name}_next_activity.csv", index=False)

# Print the DataFrame to verify
print(df)

data_processor = LogsDataProcessor(name=dataset_name, filepath="BPI_Challenge_2012.csv",  
                                    columns = ["case:concept:name", "concept:name", "time:timestamp"], #specify the columns name containing case_id, activity name and timestamp 
                                    dir_path='datasets', datetime_format="ISO8601", pool = 4)
data_processor.process_logs(task=constants.Task.NEXT_TIME, sort_temporally= True)

# Load data
data_loader = LogsDataLoader(name = dataset_name)
(train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
    vocab_size, num_output) = data_loader.load_data(constants.Task.NEXT_TIME)

# Prepare training examples for next activity prediction task
train_token_x, train_time_x, train_token_y, time_scaler, y_scaler = data_loader.prepare_data_next_time(train_df, 
                                                        x_word_dict, max_case_length)

learning_rate = 0.001
batch_size = 12
epochs = 100

# Create and train a transformer model
transformer_model = transformer.get_next_time_model(
    max_case_length=max_case_length, 
    vocab_size=vocab_size)

transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.LogCosh())

transformer_model.fit([train_token_x, train_time_x], train_token_y, 
    epochs=epochs, batch_size=batch_size)

# Initialize lists to store metrics and counts
k_list, maes, mses, rmses = [], [], [], []
num_instances_list = []

# Loop over each prefix length
for i in range(max_case_length):
    test_data_subset = test_df[test_df["k"] == i]
    num_samples = len(test_data_subset)
    
    if num_samples > 0:
        test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_next_time(
            test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False
        )
        
        y_pred = transformer_model.predict([test_token_x, test_time_x])
        _test_y = y_scaler.inverse_transform(test_y)
        _y_pred = y_scaler.inverse_transform(y_pred)
        
        mae = metrics.mean_absolute_error(_test_y, _y_pred)
        mse = metrics.mean_squared_error(_test_y, _y_pred)
        rmse = np.sqrt(mse)
        
        k_list.append(i)
        num_instances_list.append(num_samples)
        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
    else:
        # If there are no instances for this prefix, append zeros
        k_list.append(i)
        num_instances_list.append(0)
        maes.append(0)
        mses.append(0)
        rmses.append(0)

# Compute weights for each prefix
total_instances = sum(num_instances_list)
weights = [n / total_instances if total_instances > 0 else 0 for n in num_instances_list]

# Compute weighted average metrics
weighted_mae = np.average(maes, weights=num_instances_list) if total_instances > 0 else 0
weighted_mse = np.average(mses, weights=num_instances_list) if total_instances > 0 else 0
weighted_rmse = np.average(rmses, weights=num_instances_list) if total_instances > 0 else 0

# Create a DataFrame with the collected metrics
df = pd.DataFrame({
    'k': k_list,
    'weight': weights,
    'mae': maes,
    'mse': mses,
    'rmse': rmses
})

# Append the weighted averages to the DataFrame
weighted_mean_row = {
    'k': 'Weighted Mean',
    'weight': '',
    'mae': weighted_mae,
    'mse': weighted_mse,
    'rmse': weighted_rmse
}
df = pd.concat([df, pd.DataFrame([weighted_mean_row])], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv(f"{dataset_name}_next_time.csv", index=False)

# Print the DataFrame to verify
print(df)

data_processor = LogsDataProcessor(name=dataset_name, filepath="BPI_Challenge_2012.csv",  
                                    columns = ["case:concept:name", "concept:name", "time:timestamp"], #specify the columns name containing case_id, activity name and timestamp 
                                    dir_path='datasets', datetime_format="ISO8601", pool = 4)
data_processor.process_logs(task=constants.Task.REMAINING_TIME, sort_temporally= True)

# Load data
data_loader = LogsDataLoader(name = dataset_name)
(train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
    vocab_size, num_output) = data_loader.load_data(constants.Task.REMAINING_TIME)

# Prepare training examples for next activity prediction task
train_token_x, train_time_x, train_token_y, time_scaler, y_scaler = data_loader.prepare_data_remaining_time(train_df, 
                                                        x_word_dict, max_case_length)

learning_rate = 0.001
batch_size = 12
epochs = 100

# Create and train a transformer model
transformer_model = transformer.get_remaining_time_model(
    max_case_length=max_case_length, 
    vocab_size=vocab_size)

transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.LogCosh())

transformer_model.fit([train_token_x, train_time_x], train_token_y, 
    epochs=epochs, batch_size=batch_size)

# Initialize lists to store metrics and counts
k_list, maes, mses, rmses = [], [], [], []
num_instances_list = []

# Loop over each prefix length
for i in range(max_case_length):
    test_data_subset = test_df[test_df["k"] == i]
    num_samples = len(test_data_subset)
    
    if num_samples > 0:
        test_token_x, test_time_x, test_y, _, _ = data_loader.prepare_data_remaining_time(
            test_data_subset, x_word_dict, max_case_length, time_scaler, y_scaler, False
        )
        
        y_pred = transformer_model.predict([test_token_x, test_time_x])
        _test_y = y_scaler.inverse_transform(test_y)
        _y_pred = y_scaler.inverse_transform(y_pred)
        
        mae = metrics.mean_absolute_error(_test_y, _y_pred)
        mse = metrics.mean_squared_error(_test_y, _y_pred)
        rmse = np.sqrt(mse)
        
        k_list.append(i)
        num_instances_list.append(num_samples)
        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
    else:
        # If there are no instances for this prefix, append zeros
        k_list.append(i)
        num_instances_list.append(0)
        maes.append(0)
        mses.append(0)
        rmses.append(0)

# Compute weights for each prefix
total_instances = sum(num_instances_list)
weights = [n / total_instances if total_instances > 0 else 0 for n in num_instances_list]

# Compute weighted average metrics
weighted_mae = np.average(maes, weights=num_instances_list) if total_instances > 0 else 0
weighted_mse = np.average(mses, weights=num_instances_list) if total_instances > 0 else 0
weighted_rmse = np.average(rmses, weights=num_instances_list) if total_instances > 0 else 0

# Append weighted averages to the lists
k_list.append('Weighted Mean')
weights.append('')
maes.append(weighted_mae)
mses.append(weighted_mse)
rmses.append(weighted_rmse)

# Create a DataFrame with the collected metrics
df = pd.DataFrame({
    'k': k_list,
    'weight': weights,
    'mae': maes,
    'mse': mses,
    'rmse': rmses
})

# Save the DataFrame to a CSV file
df.to_csv(f"{dataset_name}_remaining_time.csv", index=False)

# Print the DataFrame to verify
print(df)