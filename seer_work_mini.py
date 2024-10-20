import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from scipy import signal


# Base directory where all samples are stored
base_directory = 'syr_data\\data\\'

# Define the channels and their respective signal names
channels_signals = {
    'Empatica-BVP': ['BVP'],
    'Empatica-EDA': ['EDA'],
    'Empatica-HR': ['HR'],
    'Empatica-TEMP': ['TEMP'],
    'Empatica-ACC': ['Acc Mag', 'Acc x', 'Acc y', 'Acc z']
}


def load_and_label_data():
    # Collect all sample directories
    sample_directories = glob.glob(os.path.join(base_directory, 'MSEL_*'))

    # Initialize lists to store all signal data, labels, and sample IDs
    all_data = []
    all_labels = []
    all_sample_ids = []

    for sample_directory in sample_directories:
        sample_name = os.path.basename(sample_directory)
        print(f"Processing sample: {sample_name}")

        # Load the label CSV file for the sample
        label_file_path = os.path.join(sample_directory, f'{sample_name}_labels.csv')
        labels_df = pd.read_csv(label_file_path)

        # Convert start and end times to datetime for easier processing
        labels_df['labels.startTime'] = pd.to_datetime(labels_df['labels.startTime'], unit='ms')
        labels_df['labels.endTime'] = labels_df['labels.startTime'] + pd.to_timedelta(labels_df['labels.duration'],
                                                                                      unit='ms')

        # Loop through each channel and signal to load data
        for channel, signals in channels_signals.items():
            channel_directory = os.path.join(sample_directory, channel)

            if not os.path.exists(channel_directory):
                continue

            for signal in signals:
                search_pattern = os.path.join(channel_directory, f"{sample_name}_{channel}_{signal}_segment_*.parquet")
                parquet_files = glob.glob(search_pattern)

                if len(parquet_files) == 0:
                    continue

                # Load each segment of the signal
                for parquet_file in parquet_files:
                    data_table = pq.read_table(parquet_file)
                    signal_data_df = data_table.to_pandas()

                    # Extract timestamp and data columns
                    timestamp_column = 'time' if 'time' in signal_data_df.columns else 'timestamp'
                    value_column = 'data' if 'data' in signal_data_df.columns else 'value'

                    # Convert timestamps to datetime
                    signal_data_df[timestamp_column] = pd.to_datetime(signal_data_df[timestamp_column], unit='ms')

                    # Initialize labels for this segment
                    segment_labels = np.zeros(len(signal_data_df))  # Assume non-seizure (0) by default

                    # Check each row in the labels_df to determine if the signal segment overlaps with a seizure
                    for _, label_row in labels_df.iterrows():
                        seizure_start = label_row['labels.startTime']
                        seizure_end = label_row['labels.endTime']

                        # Find indices where the signal data overlaps with seizure events
                        overlap_indices = signal_data_df[
                            (signal_data_df[timestamp_column] >= seizure_start) &
                            (signal_data_df[timestamp_column] <= seizure_end)
                            ].index

                        # Set these indices to '1' indicating seizure onset
                        segment_labels[overlap_indices] = 1

                    # Append the signal data, label, and sample ID to our lists
                    all_data.append(signal_data_df[value_column].values)
                    all_labels.append(segment_labels)
                    all_sample_ids.append(sample_name)

                    print(
                        f"Processed file: {os.path.basename(parquet_file)} from Sample: {sample_name}, Channel: {channel}, Signal: {signal}")

    return all_data, all_labels, all_sample_ids


# Run the data loading function
all_data, all_labels, all_sample_ids = load_and_label_data()

# Verifying the loaded data
print(f"\nTotal samples loaded: {len(all_data)}")
print(f"Shape of first signal data array: {all_data[0].shape}")
print(f"First signal data sample: {all_data[0][:5]}")
print(f"First corresponding labels sample: {all_labels[0][:5]}")

# Verify the unique sample IDs that we have loaded
unique_sample_ids = np.unique(all_sample_ids)
print(f"Unique Sample IDs: {unique_sample_ids}")
print(f"Total number of unique samples: {len(unique_sample_ids)}\n")

# Convert all_labels to a pandas Series to avoid memory issues
label_series = pd.Series(np.concatenate(all_labels[:500])) # I have set 500 to avoid excessive memory usage
# You can adjust this value based on your system's memory capacity.

# Check unique labels and their counts
unique_labels = label_series.unique()
print(f"Unique labels in the dataset: {unique_labels}")

# Count the occurrences of each label
label_counts = label_series.value_counts()
print(f"\nLabel distribution in the dataset:\n{label_counts}")

# Display a sample summary to ensure data consistency (showing the first 5 samples)
summary_df = pd.DataFrame({
    'Sample_ID': all_sample_ids[:5],  # Displaying the first 5 samples for brevity
    'Signal_Data': [data[:10] for data in all_data[:5]],  # Display first 10 signal values for each sample
    'Labels': [labels[:10] for labels in all_labels[:5]]  # Display first 10 label values for each sample
})

print("\nSummary of the first 5 loaded samples:")
print(summary_df)
#end of load data
###############################################################################
# Function to apply a 10-second sliding window and capture patient IDs
def sliding_window(data, labels, sample_id, window_size=10, fs=128):
    window_length = window_size * fs  # Total number of data points in a 10-second window
    num_windows = len(data) // window_length  # Total number of windows

    # Collect windowed data, labels, and sample IDs
    windowed_data = []
    windowed_labels = []
    windowed_sample_ids = []

    for i in range(num_windows):
        start = i * window_length
        end = start + window_length

        # Extract the window data and corresponding labels
        window_data = data[start:end]
        window_label = labels[start:end]

        # Assign a single label for the window (1 if any part is labeled as seizure, otherwise 0)
        window_label = 1 if np.any(window_label == 1) else 0

        # Append to the windowed data
        windowed_data.append(window_data)
        windowed_labels.append(window_label)
        windowed_sample_ids.append(sample_id)  # Include the sample ID for this window

    return np.array(windowed_data), np.array(windowed_labels), np.array(windowed_sample_ids)


# Apply the sliding window to our loaded data while keeping track of sample IDs
all_windowed_data = []
all_windowed_labels = []
all_windowed_sample_ids = []

for data, labels, sample_id in zip(all_data, all_labels, all_sample_ids):
    windowed_data, windowed_labels, windowed_sample_ids = sliding_window(data, labels, sample_id)
    all_windowed_data.extend(windowed_data)
    all_windowed_labels.extend(windowed_labels)
    all_windowed_sample_ids.extend(windowed_sample_ids)

# Calculate the total number of window labels and window sample IDs
total_window_labels = len(all_windowed_labels)
total_window_sample_ids = len(all_windowed_sample_ids)

# Find the unique values in window labels and sample IDs
unique_window_labels = np.unique(all_windowed_labels)
unique_window_sample_ids = np.unique(all_windowed_sample_ids)

# Display the results
print(f"Total number of window labels: {total_window_labels}")
print(f"Total number of window sample IDs: {total_window_sample_ids}")
print(f"Unique window labels: {unique_window_labels}")
print(f"Unique window sample IDs: {unique_window_sample_ids}")

print(f"Total windows created: {len(all_windowed_data)}")
print(f"Shape of the first window: {all_windowed_data[0].shape}")
print(f"First window data sample: {all_windowed_data[0][:5]}")  # Display first 5 samples of the first window
print(f"First window label: {all_windowed_labels[0]}")
print(f"First window sample ID: {all_windowed_sample_ids[0]}")


# Filtering function (remove 60 Hz noise)
def filt_data(data, fs=128):
    low = 58 / float(fs / 2)
    high = 62 / float(fs / 2)
    b, a = signal.butter(5, [low, high], btype='bandstop')
    data_filt = signal.filtfilt(b, a, data, axis=0)
    return data_filt


# Mean subtraction function
def mean_subtractor(data):
    return data - np.mean(data, axis=0)


# FFT function
def apply_fft(data):
    fft_data = np.fft.fft(data, axis=0)
    fft_magnitude = np.abs(fft_data)
    return fft_magnitude


# Apply preprocessing to each window
preprocessed_data = []

for window in all_windowed_data:
    # Apply filtering
    filtered_data = filt_data(window)

    # Apply mean subtraction
    normalized_data = mean_subtractor(filtered_data)

    # Apply FFT
    fft_data = apply_fft(normalized_data)

    preprocessed_data.append(fft_data)

total_preprocessed_data = len(preprocessed_data)

# Verify the alignment by comparing lengths
aligned_with_labels = (total_preprocessed_data == total_window_labels)
aligned_with_sample_ids = (total_preprocessed_data == total_window_sample_ids)

# Display the results
print(f"Total preprocessed windows: {total_preprocessed_data}")
print(f"Data is aligned with window labels: {aligned_with_labels}")
print(f"Data is aligned with window sample IDs: {aligned_with_sample_ids}")

# Check the shape and a sample of preprocessed data, labels, and sample IDs
print(f"Shape of the first preprocessed window: {preprocessed_data[0].shape}")
print(f"First 5 values of the preprocessed window: {preprocessed_data[0][:5]}")
print(f"Corresponding label for the first window: {all_windowed_labels[0]}")
print(f"Corresponding sample ID for the first window: {all_windowed_sample_ids[0]}")

#End of data processing
#################################################################

# Define a directory to save processed batch files
batch_data_dir = os.path.join(".", "processed_batches")
os.makedirs(batch_data_dir, exist_ok=True)

# Initialize lists for stratified sampling
seizure_windows = []
non_seizure_windows = []

unique_labels_in_windowed_data = np.unique(all_windowed_labels)
print(f"Unique labels in the windowed data: {unique_labels_in_windowed_data}")
# Separate seizure and non-seizure windows
for data, label, sample_id in zip(preprocessed_data, all_windowed_labels, all_windowed_sample_ids):
    if label == 1:
        seizure_windows.append((data, label, sample_id))
    else:
        non_seizure_windows.append((data, label, sample_id))

# Shuffle both lists to ensure randomness
np.random.shuffle(seizure_windows)
np.random.shuffle(non_seizure_windows)

# Calculate scale_pos_weight before modifying the lists
total_non_seizure = len(non_seizure_windows)
total_seizure = len(seizure_windows)

scale_pos_weight = total_non_seizure / total_seizure

# Determine the number of seizure samples to use per batch
batch_size = 5  # Adjust as needed based on memory
seizure_per_batch = min(len(seizure_windows), len(non_seizure_windows)) // batch_size

batch_count = 0

# Create balanced batches
while seizure_windows and non_seizure_windows:
    batch_data_list = []
    batch_label_list = []
    batch_patient_id_list = []

    for _ in range(seizure_per_batch):
        if seizure_windows:
            seizure_sample = seizure_windows.pop(0)
            batch_data_list.append(seizure_sample[0])
            batch_label_list.append(seizure_sample[1])
            batch_patient_id_list.append(seizure_sample[2])

        if non_seizure_windows:
            non_seizure_sample = non_seizure_windows.pop(0)
            batch_data_list.append(non_seizure_sample[0])
            batch_label_list.append(non_seizure_sample[1])
            batch_patient_id_list.append(non_seizure_sample[2])

    if len(batch_data_list) > 0:
        # Convert to NumPy arrays
        batch_data_array = np.array(batch_data_list)
        batch_labels_array = np.array(batch_label_list)
        batch_patient_ids_array = np.array(batch_patient_id_list)

        # Save the batch
        batch_file_path = os.path.join(batch_data_dir, f"batch_{batch_count}.npz")
        np.savez(batch_file_path, data=batch_data_array, labels=batch_labels_array, patient_ids=batch_patient_ids_array)
        print(f"Saved stratified batch {batch_count} to {batch_file_path}")
        batch_count += 1

print(f"\nData processing complete with stratified sampling. Data saved in batches to: {batch_data_dir}")

#end of stratified
###################################################################################
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

# Directory where processed stratified batches are saved
batch_data_dir = os.path.join(".", "processed_batches")
batch_files = glob.glob(os.path.join(batch_data_dir, "batch_*.npz"))

# Initialize lists to collect the processed data, labels, and sample IDs
all_processed_data = []
all_processed_labels = []
all_processed_sample_ids = []

# Load data from each batch file
for batch_file in batch_files:
    batch = np.load(batch_file)
    all_processed_data.extend(batch['data'])
    all_processed_labels.extend(batch['labels'])
    all_processed_sample_ids.extend(batch['patient_ids'])

# Convert lists to numpy arrays for compatibility
all_processed_data = np.array(all_processed_data)
all_processed_labels = np.array(all_processed_labels)
all_processed_sample_ids = np.array(all_processed_sample_ids)

# Convert data to sktime format for MiniROCKET
def convert_to_sktime_format(data):
    sktime_data = pd.DataFrame(pd.Series([pd.Series(window) for window in data]))
    return sktime_data

X = convert_to_sktime_format(all_processed_data)
y = all_processed_labels
groups = all_processed_sample_ids

# Initialize MiniROCKET transformer
minirocket = MiniRocketMultivariate()

# Apply Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()
fold_num = 1
results_list = []
############################################################################
# # multirocket + ridgeclassification
# for train_index, test_index in logo.split(X, y, groups=groups):
#     if len(train_index) == 0 or len(test_index) == 0:
#         print(f"Skipping fold {fold_num} due to empty train or test set.")
#         continue
#
#     print(f"Training on fold {fold_num}...")
#
#     # Train transformation on training data
#     minirocket.fit(X.iloc[train_index])
#
#     # Transform training and test data
#     X_train_transformed = minirocket.transform(X.iloc[train_index])
#     X_test_transformed = minirocket.transform(X.iloc[test_index])
#
#     # Train a classifier (RidgeClassifierCV) on transformed data
#     classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
#     classifier.fit(X_train_transformed, y[train_index])
#
#     # Evaluate the classifier on the test set
#     y_pred = classifier.predict(X_test_transformed)
#
#     # Calculate metrics
#     cm = confusion_matrix(y[test_index], y_pred)
#     print(f"Fold {fold_num} Confusion Matrix:\n{cm}")
#
#     report = classification_report(y[test_index], y_pred, target_names=['Non-Seizure', 'Seizure'], output_dict=True)
#     auc_roc = roc_auc_score(y[test_index], classifier.decision_function(X_test_transformed))
#
#     results_list.append({
#         "Fold": fold_num,
#         "Confusion Matrix": cm.tolist(),
#         "Precision (Non-Seizure)": report['Non-Seizure']['precision'],
#         "Recall (Non-Seizure)": report['Non-Seizure']['recall'],
#         "F1-Score (Non-Seizure)": report['Non-Seizure']['f1-score'],
#         "Precision (Seizure)": report['Seizure']['precision'],
#         "Recall (Seizure)": report['Seizure']['recall'],
#         "F1-Score (Seizure)": report['Seizure']['f1-score'],
#         "AUC-ROC": auc_roc
#     })
#
#     fold_num += 1
#
# # Save results to a CSV file
# results_df = pd.DataFrame(results_list)
# results_df.to_csv('minirocket_evaluation_results.csv', index=False)
# print("Results saved to 'minirocket_evaluation_results.csv'.")
#################################################
#minirocket + xgboost
import xgboost as xgb

for train_index, test_index in logo.split(X, y, groups=groups):
    if len(train_index) == 0 or len(test_index) == 0:
        print(f"Skipping fold {fold_num} due to empty train or test set.")
        continue

    print(f"Training on fold {fold_num}...")

    # Train transformation on training data
    minirocket.fit(X.iloc[train_index])

    # Transform training and test data
    X_train_transformed = minirocket.transform(X.iloc[train_index])
    X_test_transformed = minirocket.transform(X.iloc[test_index])

    # Flatten the data for compatibility with XGBoost
    X_train_flat = X_train_transformed.to_numpy()
    X_test_flat = X_test_transformed.to_numpy()

    # Convert the transformed data to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_flat, label=y[train_index])
    dtest = xgb.DMatrix(X_test_flat, label=y[test_index])

    # Define XGBoost parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": (sum(y[train_index] == 0) / sum(y[train_index] == 1)),  # Handle class imbalance
        "tree_method": "hist"
    }

    # Train the XGBoost model
    xgb_classifier = xgb.train(params, dtrain, num_boost_round=50)

    # Predict on the test set
    y_pred_prob = xgb_classifier.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate metrics
    cm = confusion_matrix(y[test_index], y_pred)
    print(f"Fold {fold_num} Confusion Matrix:\n{cm}")

    report = classification_report(y[test_index], y_pred, target_names=['Non-Seizure', 'Seizure'], output_dict=True)
    auc_roc = roc_auc_score(y[test_index], y_pred_prob)

    results_list.append({
        "Fold": fold_num,
        "Confusion Matrix": cm.tolist(),
        "Precision (Non-Seizure)": report['Non-Seizure']['precision'],
        "Recall (Non-Seizure)": report['Non-Seizure']['recall'],
        "F1-Score (Non-Seizure)": report['Non-Seizure']['f1-score'],
        "Precision (Seizure)": report['Seizure']['precision'],
        "Recall (Seizure)": report['Seizure']['recall'],
        "F1-Score (Seizure)": report['Seizure']['f1-score'],
        "AUC-ROC": auc_roc
    })

    fold_num += 1

# Save results to a CSV file
results_df = pd.DataFrame(results_list)
results_df.to_csv('minirocket_xgboost_evaluation_results.csv', index=False)
print("Results saved to 'minirocket_xgboost_evaluation_results.csv'.")

