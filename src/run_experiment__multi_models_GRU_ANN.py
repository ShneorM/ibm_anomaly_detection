import os
import random
import logging
import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef

from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU, GRU, RepeatVector
from keras.optimizers import Adam

import tensorflow as tf
import hydra
from omegaconf import DictConfig

from plotting_module import plot_results  # Import plotting function
from nab_scoring import calculate_nab_score_with_window_based_tp_fn  # Import NAB Scoring function
from anomaly_likelihood import detect_anomalies_with_likelihood  # import Anomaly Likelihood functions
from preprocessing import create_windowed_data, clean_training_data, load_and_prepare_data, add_time_features, filter_anomaly_windows

# Add additional imports or functionality below if necessary



# To ensure reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)




# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



# Define the ANN autoencoder model
def create_ann_autoencoder(input_dim, encoding_dim=14):
    """
    Creates a more powerful autoencoder model using Keras with deeper layers,
    LeakyReLU activation, batch normalization, and dropout regularization.

    Parameters:
    - input_dim: Integer, dimension of the input data.
    - encoding_dim: Integer, dimension of the encoded representation.

    Returns:
    - autoencoder: Keras Model, compiled autoencoder model.
    """
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(128)(input_layer)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)

    encoded = Dense(64)(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)

    encoded = Dense(encoding_dim)(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)

    # Decoder
    decoded = Dense(64)(encoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)

    decoded = Dense(128)(decoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)

    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return autoencoder



# Define the updated GRU autoencoder model
def create_gru_autoencoder(input_dim, time_steps=1, encoding_dim=14, layer_cnt=7, cell_cnt=16, dropout_rate=0.2):
    """
    Creates a GRU autoencoder model using Keras with multiple layers and dropout.

    Parameters:
    - input_dim: Integer, dimension of the input data.
    - time_steps: Integer, number of time steps for the sequential data.
    - encoding_dim: Integer, dimension of the encoded representation.
    - layer_cnt: Integer, number of GRU layers to stack.
    - cell_cnt: Integer, number of units (cells) in each GRU layer.
    - dropout_rate: Float, dropout rate to be applied for regularization.

    Returns:
    - gru_autoencoder: Keras Model, compiled GRU autoencoder model.
    """
    input_layer = Input(shape=(time_steps, input_dim))

    # Encoder with multiple GRU layers and dropout
    x = input_layer
    for _ in range(layer_cnt):
        x = GRU(cell_cnt, activation='relu', return_sequences=True, recurrent_dropout=dropout_rate)(x)

    # Final encoding layer
    encoded = GRU(encoding_dim, activation='relu')(x)

    # Decoder with RepeatVector and GRU layer
    decoded = RepeatVector(time_steps)(encoded)
    for _ in range(layer_cnt):
        decoded = GRU(cell_cnt, activation='relu', return_sequences=True)(decoded)

    # Output layer (reconstruction)
    output_layer = GRU(input_dim, return_sequences=True, activation='relu')(decoded)

    gru_autoencoder = Model(input_layer, output_layer)
    gru_autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return gru_autoencoder



# Train the autoencoder model
def train_autoencoder(X_train, model_type='ANN', encoding_dim=14, epochs=50, batch_size=32, validation_split=0.2):
    """
    Trains the autoencoder model (ANN or GRU) using the provided training data.

    Parameters:
    - X_train: Array, training data.
    - encoding_dim: Integer, dimension of the encoded representation.
    - epochs: Integer, number of epochs for training.
    - batch_size: Integer, batch size for training.
    - validation_split: Float, fraction of data to be used for validation.
    - model_type: String, type of model to train ('ANN' or 'GRU').

    Returns:
    - autoencoder: Keras Model, trained autoencoder model.
    - history: Keras History object, training history.
    """
    input_dim = X_train.shape[1]
    print('model_type: ', model_type)

    if model_type == 'GRU':
        time_steps = 1  # Adjust the time steps to 1
        X_train_windowed, missing_points = create_windowed_data(X_train, time_steps)

        # Update input_dim for GRU
        input_dim = X_train_windowed.shape[2]

        if model_type == 'GRU':
            # autoencoder = create_gru_autoencoder(input_dim, time_steps, encoding_dim)
            autoencoder = create_gru_autoencoder(input_dim, time_steps, encoding_dim)


        print("Model Summary: ")
        autoencoder.summary()

        # Train the model
        history = autoencoder.fit(X_train_windowed, X_train_windowed, epochs=epochs, batch_size=batch_size,
                                  validation_split=validation_split,
                                  shuffle=False, verbose=0)

        # Set the reconstruction error for the missing initial points to zero
        recon_errors = np.zeros((missing_points, input_dim))  # Reconstruction error of zero for missing points

        return autoencoder, history #, recon_errors

    else:
        # If it's not GRU, train a standard ANN autoencoder
        # autoencoder = create_autoencoder(input_dim, encoding_dim)
        autoencoder = create_ann_autoencoder(input_dim, encoding_dim)

        print("Model Summary: ")
        autoencoder.summary()

        # Train the model
        history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                                  validation_split=validation_split,
                                  shuffle=False, verbose=0)

        # plot_training_history(history)

        return autoencoder, history



# Plot training history
def plot_training_history(history):
    """
    Plots the training and validation loss over epochs.

    Parameters:
    - history: Keras History object, contains the training and validation loss values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



# Evaluate performance metrics
def evaluate_performance(y_true, y_pred):
    """
    Evaluates the performance of the anomaly detection using various classification metrics.

    Parameters:
    - y_true: Array, true labels.
    - y_pred: Array, predicted labels.

    Returns:
    - precision: Float, precision of the model.
    - recall: Float, recall of the model.
    - f1: Float, F1-score of the model.
    - accuracy: Float, accuracy of the model.
    - conf_matrix: Array, confusion matrix.
    - mcc: Float, Matthews correlation coefficient.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return precision, recall, f1, accuracy, conf_matrix, mcc



def save_results_to_csv(results_df, result_directory, use_weights):
    # Define a file path variable
    filename_prefix = 'weighted_' if use_weights else 'unweighted_'
    csv_filepath = os.path.join(result_directory, f'{filename_prefix}results.csv')

    # Save the results DataFrame to CSV using the defined file path
    results_df.to_csv(csv_filepath, index=False)

    # Optionally return the file path if needed elsewhere
    return csv_filepath


# Grid Search for optimal parameters using weighted NAB score
def grid_search(train_data, test_data, anomaly_windows, anomaly_windows_test, result_directory, model_save_directory, nab_scoring_profile, use_weights=False, model='GRU'):
    """
    Perform a grid search to find the best parameters for anomaly detection.

    Parameters:
    - train_data: DataFrame, training data without anomalies.
    - test_data: DataFrame, testing data including anomalies.
    - anomaly_windows: DataFrame, training anomaly windows for reference.
    - anomaly_windows_test: DataFrame, testing anomaly windows for evaluation.
    - result_directory: String, path to save results.
    - model_save_directory: String, directory to save trained models.
    - nab_scoring_profile: String, NAB scoring profile ('standard' or 'reward_fn').
    - use_weights: Boolean, whether to use weighted NAB scoring.
    - model: String, model type ('GRU' or 'ANN').

    Returns:
    - best_params: Tuple, best combination of threshold, long window, and short window.
    - best_score: Float, best normalized NAB score.
    - best_results: Dict, detailed results for the best combination.
    - csv_filepath: String, path to the CSV file with results.
    """

    # Define the parameter grid
    # anomaly_threshold_values = [0.9990, 0.9995, 0.9996, 0.9997, 0.9998]
    # long_window_values = [20, 25, 30, 35, 40, 50]
    # short_window_values = [1, 2, 3, 4, 5]

    anomaly_threshold_values = [0.9996]
    long_window_values = [30]
    short_window_values = [2]

    best_score = -float('inf')
    best_params = None
    best_results = None

    all_results = []

    # Iterate over all combinations of anomaly_threshold, long_window, and short_window
    for anomaly_threshold, long_window, short_window in itertools.product(anomaly_threshold_values, long_window_values, short_window_values):
        log.info(f"Running experiment with Threshold={anomaly_threshold}, Long Window={long_window}, Short Window={short_window}")

        # Run the experiment for this combination
        precision, recall, f1, accuracy, conf_matrix, mcc, is_anomalies, likelihoods, results_df, raw_nab_score, normalized_nab_score, detection_counters_nab = run_experiment(
            train_data, test_data, anomaly_windows, anomaly_windows_test, model_save_directory, nab_scoring_profile,
            anomaly_threshold=anomaly_threshold,
            long_window=long_window,
            short_window=short_window,
            use_weights=use_weights,
            model=model
        )

        # Log NAB score
        log.info(f"Raw NAB Score: {raw_nab_score}, Normalized NAB Score: {normalized_nab_score}")
        print("++++++detection_counters_nab: ", detection_counters_nab)

        # Collect results
        detection_counters = {
            'tp': conf_matrix[1, 1],
            'tn': conf_matrix[0, 0],
            'fp': conf_matrix[0, 1],
            'fn': conf_matrix[1, 0],
            # 'issue_detected': anomaly_windows[anomaly_windows['anomaly_source'] == 1].shape[0],
            # 'im_detected': anomaly_windows[anomaly_windows['anomaly_source'] == 2].shape[0],
            # 'TestLog_detected': anomaly_windows[anomaly_windows['anomaly_source'] == 3].shape[0],
        }

        # Use detection counters from run_experiment or add it separately if needed
        result = {
            'Threshold': anomaly_threshold,
            'Long Window': long_window,
            'Short Window': short_window,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'MCC': mcc,
            # 'TP': detection_counters['tp'],
            'TP': detection_counters_nab['tp'],
            'TP_Org': detection_counters['tp'],
            'FN_Org': detection_counters['fn'],
            'TN': detection_counters['tn'],
            'FP': detection_counters['fp'],
            'FN': detection_counters_nab['fn'], # It should window based; not point based
            'Raw NAB Score': raw_nab_score,
            'Normalized NAB Score': normalized_nab_score,
            'Problem Detected': detection_counters_nab['issue_detected'],
            'Slack Detected': detection_counters_nab['im_detected'],
            'Incident Detected': detection_counters_nab['TestLog_detected']
        }

        all_results.append(result)

        # Check if this is the best score so far based on normalized NAB score
        if normalized_nab_score > best_score:
            best_score = normalized_nab_score
            best_params = (anomaly_threshold, long_window, short_window)
            best_results = result

    # Ensure the directory exists
    dir_to_save_to = os.path.dirname(result_directory)
    if dir_to_save_to and not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)

    results_df_all = pd.DataFrame(all_results)
    # Define a file path variable
    filename_prefix = 'weighted_' if use_weights else 'unweighted_'
    csv_filepath = os.path.join(result_directory, f'{filename_prefix}_{model}_results.csv')
    results_df_all.to_csv(csv_filepath, index=False)

    # Save results to CSV
    unweighted_csv_filepath = os.path.join(result_directory, (filename_prefix + '_' + model + '_anomaly_detection_results.csv'))
    results_df.to_csv(unweighted_csv_filepath, index=True)

    log.info(f"Best Params: {best_params}")
    log.info(f"Best Raw NAB Score: {best_results['Raw NAB Score']}")
    log.info(f"Best Normalized NAB Score: {best_score}")


    # Plot the result
    plot_results(
        results_df=results_df,
        is_anomalies=is_anomalies,
        # anomaly_windows=anomaly_windows,
        anomaly_windows=anomaly_windows_test,
        result_directory=result_directory,
        file_name_prefix='unweighted_',
        raw_nab_score=raw_nab_score,  # Pass raw NAB score here
        normalized_nab_score=normalized_nab_score,  # Pass normalized NAB score here
        model=model
    )

    return best_params, best_score, best_results, csv_filepath


def run_experiment(train_data, test_data, anomaly_windows, anomaly_windows_test, model_save_directory, nab_scoring_profile, anomaly_threshold=0.9996, long_window=30, short_window=2, use_weights=False, model='GRU'):
    """
    Run anomaly detection for a single parameter combination.

    Parameters:
    - train_data: DataFrame, cleaned training data.
    - test_data: DataFrame, test data for evaluation.
    - anomaly_windows: DataFrame, training anomaly windows.
    - anomaly_windows_test: DataFrame, testing anomaly windows.
    - model_save_directory: String, directory to save trained models.
    - nab_scoring_profile: String, NAB scoring profile ('standard' or 'reward_fn').
    - anomaly_threshold: Float, threshold for anomaly detection.
    - long_window: Integer, length of the long window for NAB scoring.
    - short_window: Integer, length of the short window for NAB scoring.
    - use_weights: Boolean, use weighted NAB scoring.
    - model: String, model type ('AE' or 'GRU').

    Returns:
    - precision, recall, f1, accuracy, conf_matrix, mcc: Evaluation metrics.
    - is_anomalies, likelihoods, results_df: Detection results.
    - raw_nab_score, normalized_nab_score: NAB scores.
    - detection_counters: Count of true positives, false negatives, etc.
    """


    log.info(f"Running experiment with params: Threshold={anomaly_threshold}, Long Window={long_window}, Short Window={short_window}")

    # Ensure anomaly windows are timezone-naive
    anomaly_windows['anomaly_window_start'] = anomaly_windows['anomaly_window_start'].dt.tz_localize(None)
    anomaly_windows['anomaly_window_end'] = anomaly_windows['anomaly_window_end'].dt.tz_localize(None)

    # Clean training data (excluding anomaly windows)
    cleaned_training_data = clean_training_data(train_data, anomaly_windows)

    # Scale training data
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(cleaned_training_data)

    # Train AE or GRU autoencoder
    model_type = model #'GRU' or 'ANN' as needed
    print("model_type:: ", model_type)


    # Train the autoencoder based on model type
    # Define the path to the trained models directory
    trained_models_dir = model_save_directory
    os.makedirs(trained_models_dir, exist_ok=True)  # Ensure the directory exists

    # Define the model filename based on the model type
    model_filename = f"{trained_models_dir}{model_type}_autoencoder.h5"

    # Check if the model file exists
    if os.path.exists(model_filename):
        print(f"Loading trained model: {model_filename}")
        autoencoder = load_model(model_filename)
    else:
        print(f"No trained model found for {model_type}. Training a new model...")
        # Train the autoencoder
        autoencoder, _ = train_autoencoder(scaled_train_data, model_type=model_type)

        # Save the trained model
        autoencoder.save(model_filename)
        print(f"Model saved: {model_filename}")


    # Scale test data
    scaled_test_data = scaler.transform(test_data)

    # Detect anomalies using reconstruction error and likelihood based on model type
    # if model_type == 'GRU' or 'LSTM':
    if model_type == 'GRU':
        # Detect anomalies using reconstruction error and likelihood for GRU
        is_anomalies, likelihoods, reconstruction_error = detect_anomalies_with_likelihood(
            autoencoder, scaled_test_data, anomaly_threshold, long_window, short_window, test_data.index,
            model_type=model_type, time_steps=1
        )
    else:
        # Detect anomalies using reconstruction error and likelihood for AE
        is_anomalies, likelihoods, reconstruction_error = detect_anomalies_with_likelihood(
            autoencoder, scaled_test_data, anomaly_threshold, long_window, short_window, test_data.index,
            model_type=model_type
        )


    # Create 'true_anomaly' labels based on the test anomaly windows anomaly_windows_test (removed incidents)
    y_true = [1 if ((anomaly_windows_test['anomaly_window_start'] <= t) & (t <= anomaly_windows_test['anomaly_window_end'])).any() else 0 for t in test_data.index]

    # Create results DataFrame for evaluation
    results_df = pd.DataFrame({
        '5XX_count': test_data.sum(axis=1),  # Adjust as needed for your data
        'true_anomaly': y_true,  # This is what the function expects
        'predicted_anomaly': is_anomalies,  # The output of the model
        'anomaly_likelihood': likelihoods,
        'reconstruction_error': reconstruction_error
    })

    print("SAMPLE RESULT DF: ", results_df.head())

    # Evaluate performance based on ground truth
    precision, recall, f1, accuracy, conf_matrix, mcc = evaluate_performance(y_true, is_anomalies)

    log.info(f"Experiment results - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}, ConfusionMatrix: {conf_matrix}")

    # Calculate the weighted NAB score and normalized NAB score
    raw_nab_score, normalized_nab_score, false_positive_count, false_negative_count, detection_counters = calculate_nab_score_with_window_based_tp_fn(
        results_df, anomaly_windows_test, nab_scoring_profile, true_col='true_anomaly', pred_col='predicted_anomaly'
    )


    # Return all values needed for grid_search
    return precision, recall, f1, accuracy, conf_matrix, mcc, is_anomalies, likelihoods, results_df, raw_nab_score, normalized_nab_score, detection_counters





@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to coordinate anomaly detection using autoencoders.

    Steps:
    1. Load and preprocess the data.
    2. Extract training and testing windows.
    3. Prepare ground truth anomaly windows for evaluation.
    4. Perform grid search to optimize anomaly detection parameters.

    Parameters:
    - cfg: DictConfig, configuration object containing paths, parameters, and settings.
    """
    log.info("Starting main function")

    # Load and prepare data
    df_prodlive, gt_utc_df = load_and_prepare_data(cfg)
    df_prodlive = add_time_features(df_prodlive)

    # Extract experiment parameters
    start_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.start_date)
    train_end_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.train_end_date)
    test_start_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.test_start_date)
    end_date = pd.Timestamp(cfg.train_test_config.experiment_parameters.end_date) + timedelta(days=1)
    result_directory = cfg.train_test_config.result_file.path
    model_save_directory = cfg.train_test_config.model_save_path
    nab_scoring_profile = cfg.evaluation.nab_scoring_profile
    use_model = cfg.train_test_config.use_model




    # Filter data for training and testing
    working_data = df_prodlive.loc[start_date:end_date]
    train_data = working_data.loc[start_date:train_end_date]
    test_data = working_data.loc[test_start_date:end_date]

    # Prepare anomaly windows
    anomaly_windows, anomaly_windows_test = filter_anomaly_windows(gt_utc_df, start_date, end_date, test_start_date)
    log.info(
        f"Anomaly windows prepared. Training windows: {len(anomaly_windows)}, Testing windows: {len(anomaly_windows_test)}")

    # Call grid search or other functions
    log.info("Starting Grid Search for best parameters...")
    best_params_unweighted, best_unweighted_score, best_results_unweighted, result_csv_filepath = grid_search(
        train_data, test_data, anomaly_windows, anomaly_windows_test, result_directory, model_save_directory, nab_scoring_profile, use_weights=False,
        model=use_model
    )
    log.info(f"{use_model} models's best NAB score with {nab_scoring_profile} profile: {best_unweighted_score}")




if __name__ == "__main__":
    main()
