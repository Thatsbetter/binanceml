## Overview
This Python project aims to predict cryptocurrency prices using deep learning. It employs TensorFlow and Keras libraries to build a recurrent neural network (RNN) with LSTM (Long Short-Term Memory) layers. The project is principally intended for educational purposes to demonstrate the application of RNNs in financial predictions.

## Libraries and Tools Used
- TensorFlow & Keras: For constructing and training the deep learning model.
- Pandas: To manipulate and ingest data.
- NumPy: For handling numerical operations.
- Scikit-Learn: Used for scaling and preprocessing data.
- os and Time: For managing file paths and timing events.

## Data
The dataset, presumably named `BTTUSDT-formated.csv`, should feature a 'price' column, representing the cryptocurrency's price, which is critical for training the model.

## Features of the Code

Data Preprocessing
- Feature Construction: Future price prediction columns are generated.
- Normalization: Application of percentage change and scaling to normalize the dataset.
- Splitting: Data is divided into training and validation sets.

Model Architecture
- Layers: Three LSTM layers are used, interspersed with Dropout layers and Batch Normalization to prevent overfitting.
- Output: A softmax activation function determines buy or sell signals from two classes.

Training
- Optimizer: Adam optimizer minimizes sparse categorical cross-entropy.
- Callbacks: Include model checkpoints to save the best models based on validation accuracy.

Validation and Testing
- Post-training evaluation uses a separate validation set.

## Usage

Environment Setup
bash
# Ensure Python 3.x and required packages are installed
pip install tensorflow pandas numpy scikit-learn


Running the Code
- Store your dataset in the designated directory (`DataSet/`).
- Adjust parameters like `SEQ_LENGTH`, `FUTURE_PRIDICT`, as needed for your dataset.
- Execute the script to start training the model and evaluate its performance.

Output
- Model configurations and weights are saved in the `models/` directory.
- Real-time metrics for training and validation are displayed during execution.
- Final loss and accuracy metrics are provided after testing on the validation data.

## Model Performance and Improvements
- Monitor test loss and accuracy to assess the model.
- Hyperparameter tuning (e.g., `EPOCHS`, `BATCH_SIZE`) may enhance performance.
- Consider more complex LSTM structures or enriching the dataset to reduce overfitting.

## Disclaimer
This model is for educational use only and not for actual trading. Financial decisions should not be based on this model due to the inherent volatility of cryptocurrency markets.

## Note
Ensure data formatting aligns with the script's requirements, especially in terms of sequence generation for LSTM input. Adjustments may be necessary depending on the original dataset's structure.

This README format structures information logically for clarity, guiding the user from an overview to detailed instructions and considerations for using the cryptocurrency prediction model efficiently.
