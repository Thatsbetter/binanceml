Overview
This Python project is designed to predict cryptocurrency prices using deep learning techniques. The code leverages the TensorFlow and Keras libraries to construct a recurrent neural network (RNN) that uses LSTM (Long Short-Term Memory) layers to process sequential data for the prediction of cryptocurrency price movements.

Libraries and Tools Used
- TensorFlow & Keras: To build and train the deep learning model.
- Pandas: For data manipulation and ingestion.
- NumPy: For numerical operations.
- Scikit-Learn: For preprocessing data scales.

Data
The model is trained on a dataset assumed to be stored under `DataSet/BTTUSDT-formated.csv`. This dataset should have at least one column named `price` which represents the price of the cryptocurrency.

Features of the Code
1. Data Preprocessing:
   - Future price columns and targeted price shifts are created.
   - The dataset is split into a training set and a validation set.
   - Data normalization and scaling are applied to assist effective model training.

2. Model Architecture:
   - The network consists of three LSTM layers, complemented with Dropout and Batch Normalization to prevent overfitting and ensure faster convergence.
   - The final output is processed through softmax activation function in the dense layer for classification into two classes (buy and sell signals).

3. Training:
   - The model is trained using the Adam optimizer, with a focus on minimizing the sparse categorical cross-entropy loss.
   - The training process is managed with callbacks including tensor checkpoints for model saving based on validation accuracy.

4. Validation and Testing:
   - Post training, the model is evaluated using the separate validation dataset to ensure that it generalizes well to new data.

Usage

1. Environment Setup:
   - Ensure that Python 3.x is installed along with the libraries mentioned.
   - Install the required packages using pip:

     
     pip install tensorflow pandas numpy scikit-learn
     

2. Running the Code:
   - Place the dataset in the proper directory as mentioned (`DataSet/BTTUSDT-formated.csv`).
   - Adjust the sequence length, future prediction offset, and other parameters as required.
   - Run the script to begin training and subsequently evaluate the model.

3. Output:
   - Model weights and configurations are saved in the `models` directory configured in the script.
   - During run-time, training and validation metrics are printed to track progress.
   - Final metrics including loss and accuracy are outputted after evaluation on the validation set.

Model Performance and Improvements
- Monitor the "Test loss" and "Test accuracy" metrics to evaluate model performance.
- For better performance, consider tuning hyperparameters such as `EPOCHS`, `BATCH_SIZE`, and learning rate.
- More advanced LSTM configurations or adding more data might help in reducing overfitting and improving prediction accuracy.

Note
This script assumes that proper data formatting and preprocessing are handled in the input CSV file. Ensure that the data provided correctly aligns with these expectations, particularly in terms of the sequence generation for LSTM input.

Disclaimer
This model and its code serve as a basic demonstration for educational purposes only. Cryptocurrency markets are highly volatile, and financial investments should be made carefully and not based on this model’s output alone.
