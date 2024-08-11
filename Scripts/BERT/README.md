# TABSA as a Sentence Pair Classification Task Using BERT

## Overview
This repository provides an implementation for Targeted Aspect-Based Sentiment Analysis (TABSA), framed as a sentence pair classification task using BERT. The model is applied to the TelecomSent dataset (formerly SentiTel) to classify sentiments and aspects within sentences related to telecom services.

## Requirements
Ensure your environment is set up with the following dependencies:

- **PyTorch**: 1.0.0
- **Python**: 3.7.1
- **TensorFlow**: 1.13.1 (needed only for converting BERT TensorFlow models to PyTorch)
- **NumPy**: 1.15.4
- **NLTK**
- **Scikit-learn**

## Step 1: Datasets
The datasets are located in the `TelecomSent/Code/BERT/Data/` directory.

| File        | Description                                          |
|-------------|------------------------------------------------------|
| `train.tsv` | Training set used for model training                 |
| `dev.tsv`   | Validation set used for tuning hyperparameters       |
| `test.tsv`  | Test set used for evaluating the final model         |

## Step 2: Prepare BERT PyTorch Model

Download [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert) and convert the TensorFlow checkpoint to a PyTorch model using the following command:

```
python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
  --pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```
## Step 3: Train the Model
Train the BERT model on the TelecomSent dataset with the following command:
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_TABSA.py \
```
  --task_name sentihood_NLI_M \
  --data_dir data/TelecomSent/bert-pair/ \
  --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
  --eval_test \
  --do_lower_case \
  --max_seq_length 128 \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 6.0 \
  --output_dir results/TelecomSent/NLI_M \
  --seed 42
```  
## Step 4: Evaluate the Model
Evaluate the trained model's performance on the test set by calculating Accuracy, F1 score, and AUC using the following command:
python evaluation.py --task_name TelecomSent_NLI_M --pred_data_dir results/TelecomSent/NLI_M/test_ep_4.txt
This evaluation will produce the metrics needed to assess the model's effectiveness on the TelecomSent dataset.
