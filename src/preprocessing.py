import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_and_prepare_data(cfg):
    """
    Load and prepare the 5XX features data and anomaly windows.

    Parameters:
    - cfg: DictConfig, configuration object.

    Returns:
    - df_prodlive: DataFrame, prepared 5XX features data.
    - gt_utc_df: DataFrame, anomaly windows data.
    """
    input_filename = cfg.train_test_config.input_files.observations_file_name
    gt_input_filename = cfg.train_test_config.input_files.ground_truth_labels_file_name
    minutes_before = cfg.train_test_config.anomaly_window.minutes_before

    # Load the 5XX features only
    if os.path.exists(input_filename):
        filtered_df = pd.read_csv(input_filename, index_col='interval_start')
        log.info(f"File '{input_filename}' loaded successfully.")
    else:
        log.info(f"File '{input_filename}' not found. Generating data using 'prepare_5xx_features'.")
        filtered_df = prepare_5xx_features(cfg)

    filtered_df.fillna(0, inplace=True)
    filtered_df.index = pd.to_datetime(filtered_df.index)

    # Load anomaly windows
    gt_df = pd.read_csv(gt_input_filename)
    gt_df['anomaly_start'] = pd.to_datetime(gt_df['anomaly_start'], utc=True)
    gt_df['anomaly_end'] = pd.to_datetime(gt_df['anomaly_end'], utc=True)

    gt_df['anomaly_window_start'] = gt_df['anomaly_start'] - pd.Timedelta(minutes=minutes_before)
    gt_df['anomaly_window_end'] = gt_df['anomaly_end']

    gt_utc_df = gt_df[['number', 'anomaly_window_start', 'anomaly_window_end', 'anomaly_source']].copy()

    return filtered_df, gt_utc_df


def prepare_5xx_features(cfg):
    """
    Prepare 5XX features from a parquet file. Filters columns, sums them up, and saves the result to a CSV file.

    :param cfg: Configuration object containing the input and output file paths and filename template.
    """
    # Construct input and output paths
    input_path = os.path.join(cfg.data_preparation_pipeline.features_5xx_prep.input_file.path,
                              cfg.data_preparation_pipeline.features_5xx_prep.input_file.filename)
    output_path = os.path.join(cfg.data_preparation_pipeline.features_5xx_prep.output_file.path,
                               cfg.data_preparation_pipeline.features_5xx_prep.output_file.filename_template)

    # Inspect the schema of the Parquet file to identify columns
    parquet_file = pq.ParquetFile(input_path)
    all_columns = parquet_file.schema.names

    # Filter columns using regex and include 'interval_start'
    regex_pattern = cfg.data_preparation_pipeline.features_5xx_prep.regex_pattern
    columns_to_keep = [col for col in all_columns if
                       re.search(regex_pattern, col)]
    if 'interval_start' in all_columns and 'interval_start' not in columns_to_keep:
        columns_to_keep.append('interval_start')

    # Load only the filtered columns
    filtered_df = pd.read_parquet(input_path, columns=columns_to_keep)

    # Convert 'interval_start' to datetime and set it as index
    if 'interval_start' in filtered_df.columns:
        filtered_df['interval_start'] = pd.to_datetime(filtered_df['interval_start'], unit="s")
        filtered_df.set_index('interval_start', inplace=True)


    # Extract data center name from filename
    match = re.search(r'pivoted_data_([^_]+)\.parquet', cfg.data_preparation_pipeline.features_5xx_prep.input_file.filename)
    data_center_name = match.group(1) if match else 'unknown'

    # Construct full output file path
    output_file_location = output_path.format(data_center_name=data_center_name)

    # Ensure the directory exists
    dir_to_save_to = os.path.dirname(output_file_location)
    if dir_to_save_to and not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)

    # Save to CSV
    filtered_df.to_csv(output_file_location, index=True)
    log.info(f'Data saved as "{output_file_location}"')

    return filtered_df



def add_time_features(df):
    """
    Add time-related features (sine and cosine transformations) to the DataFrame.

    Parameters:
    - df: DataFrame, input data.

    Returns:
    - df: DataFrame, updated with time-related features.
    """
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear

    # Hourly seasonality
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Daily seasonality (weekly pattern)
    df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Drop intermediate columns if not needed
    df.drop(columns=['hour', 'minute', 'day_of_week', 'day_of_year'], inplace=True)
    return df


def filter_anomaly_windows(gt_utc_df, start_date, end_date, test_start_date):
    """
    Filter and prepare anomaly windows for training and testing.

    Parameters:
    - gt_utc_df: DataFrame, original anomaly windows.
    - start_date: Timestamp, start date of the experiment.
    - end_date: Timestamp, end date of the experiment.
    - test_start_date: Timestamp, start date of the testing period.

    Returns:
    - anomaly_windows: DataFrame, training anomaly windows.
    - anomaly_windows_test: DataFrame, testing anomaly windows.
    """
    gt_utc_df['anomaly_window_start'] = gt_utc_df['anomaly_window_start'].dt.tz_localize(None)
    gt_utc_df['anomaly_window_end'] = gt_utc_df['anomaly_window_end'].dt.tz_localize(None)

    # Training anomaly windows
    anomaly_windows = gt_utc_df.loc[
        (gt_utc_df['anomaly_window_start'] >= start_date) &
        (gt_utc_df['anomaly_window_end'] <= end_date)
        ]

    # Testing anomaly windows
    anomaly_windows_test = anomaly_windows.loc[
        (anomaly_windows['anomaly_window_start'] >= test_start_date) &
        (anomaly_windows['anomaly_window_end'] <= end_date)
        ]

    return anomaly_windows, anomaly_windows_test


def create_windowed_data(X, time_steps):
    """
    Creates windowed data from the input array with specified time steps.

    Parameters:
    - X: Array, input data.
    - time_steps: Integer, number of time steps for the sliding window.

    Returns:
    - X_windowed: Array, the windowed input data.
    - missing_points: Integer, number of missing points due to windowing.
    """
    print("We are at the windowing step: ", X.shape)
    num_samples = X.shape[0] - time_steps + 1
    X_windowed = []

    for i in range(num_samples):
        X_windowed.append(X[i:i + time_steps])

    X_windowed = np.array(X_windowed)
    missing_points = time_steps - 1
    print("Done windowing step: ", X_windowed.shape)

    return X_windowed, missing_points


def clean_training_data(df, anomaly_windows):
    """
    Removes anomalies from the training data based on ground truth anomaly windows.

    Parameters:
    - df: DataFrame, training data.
    - anomaly_windows: DataFrame, contains the start and end times of anomaly windows.

    Returns:
    - df_cleaned: DataFrame, cleaned training data without anomalies.
    """
    # Convert 'anomaly_window_start' and 'anomaly_window_end' to timezone-naive
    anomaly_windows['anomaly_window_start'] = anomaly_windows['anomaly_window_start'].dt.tz_localize(None)
    anomaly_windows['anomaly_window_end'] = anomaly_windows['anomaly_window_end'].dt.tz_localize(None)

    df_label = df.copy()
    df_label['5XX_count'] = df_label.sum(axis=1)
    df_label['isAnomaly'] = 0

    for index, row in df_label.iterrows():
        # Ensure that index is tz-naive before comparison
        is_anomaly = ((anomaly_windows['anomaly_window_start'] <= index) &
                      (index <= anomaly_windows['anomaly_window_end'])).any()
        df_label.at[index, 'isAnomaly'] = 1 if is_anomaly else 0

    df_cleaned = df_label[df_label['isAnomaly'] == 0].drop(columns=['isAnomaly', '5XX_count'])

    return df_cleaned
