# Running Attentive LSTM Models for Sentiment Analysis

This project implements an Attentive LSTM model for sentiment analysis on target-aspect pairs. The model classifies reviews into "Positive", "Negative", or "None" categories. After training and evaluating models for each target-aspect pair, the results are averaged to calculate Strict accuracy, Macro-F1, and Sentiment accuracy metrics.

### Training Process

To train the model for each target-aspect pair:

1. Obtain the target-aspect pair data files.
2. Copy the following files to the target-aspect folder:
   - `models` folder
   - `data_utils.py`
   - `download_glove.py`
   - `evaluating_LSTM.py`
   - `test.py`
   - `train.py`
3. Execute the `train.py` script to train the model for that specific target-aspect pair.

### Customizable Hyperparameters

Before running the models, you can adjust the following parameters:

| Parameter       | Description                  | Type  | Default Value                    | Options        |
|-----------------|------------------------------|-------|----------------------------------|----------------|
| train_tsv       | Training TSV file path       | str   | "data/train_dev_data_oversampled.tsv" | -              |
| model           | Model type                   | str   | "att"                            | "naive" or "att" |
| glove           | Use GloVe embeddings         | bool  | -                                | -              |
| embedding_size  | Word embedding dimensions    | int   | 200                              | 50, 100, 200, 300 |
| num_hidden      | RNN hidden layer size        | int   | 100                              | -              |
| num_layers      | RNN depth                    | int   | 2                                | -              |
| learning_rate   | Learning rate                | float | 1e-3                             | -              |
| batch_size      | Batch size                   | int   | 64                               | -              |
| num_epochs      | Number of training epochs    | int   | 10                               | -              |
| keep_prob       | Dropout keep probability     | float | 0.8                              | -              |
| checkpoint_dir  | Model checkpoint directory   | str   | "saved_model"                    | -              |

### Testing

To evaluate the model's classification accuracy on test data, run the `test.py` script with the `test_tsv` parameter set to the test dataset path:

```bash
python test.py --test_tsv="/data/test.tsv"
```

### Sample Results

Here are example results when trained and tested with default hyperparameters:

| Metric          | mtn_calls | mtn_data | mtn_general | mtn_network | Average |
|-----------------|-----------|----------|-------------|-------------|---------|
| sentiment_Acc   | 0.847     | 0.788    | 0.648       | 0.806       | **0.772** |
| aspect_Macro_F1 | 0.089     | 0.303    | 0.349       | 0.055       | **0.199** |

