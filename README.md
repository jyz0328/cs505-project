# Semantic Similarity Detection Project

This repository contains the resources and results of our **Semantic Similarity Detection** project. In this project, we evaluate traditional models such as **Bag-of-Words (BoW)** and **TF-IDF**, using the **STS Benchmark (STSb)** dataset. We measure the performance of these models using **Cosine Similarity**, **Pearson Correlation**, and **Spearman's Rank Correlation**.

## Repository Contents

### Dataset Files

- **`stsb_train.csv`**: Training set from the STS Benchmark dataset, used to train the models. Each row contains:
  - `sentence1`: First sentence in the pair.
  - `sentence2`: Second sentence in the pair.
  - `score`: Human-annotated similarity score (normalized between 0 and 1).

- **`stsb_test.csv`**: Test set from the STS Benchmark dataset, used to evaluate the models on unseen data.

- **`stsb_validation.csv`**: Validation set from the STS Benchmark dataset, used to tune and validate the models during training.

### Processed Data

- **`processed_train.csv`**: Contains the processed results after applying the **BoW** and **TF-IDF** models to the training data (`stsb_train.csv`). It includes the original columns (`sentence1`, `sentence2`, `score`), along with additional columns:
  - `BoW score`: Predicted similarity score from the BoW model.
  - `TF-IDF score`: Predicted similarity score from the TF-IDF model.
  - `BoW Pearson Correlation`: Pearson correlation coefficient for BoW (only on the first row).
  - `BoW Spearman Rank Correlation`: Spearman rank correlation coefficient for BoW (only on the first row).
  - `TF-IDF Pearson Correlation`: Pearson correlation coefficient for TF-IDF (only on the first row).
  - `TF-IDF Spearman Rank Correlation`: Spearman rank correlation coefficient for TF-IDF (only on the first row).
  - `BoW/True`: Ratio of BoW predicted score to the actual similarity score.
  - `TF-IDF/True`: Ratio of TF-IDF predicted score to the actual similarity score.
    
### Usage

- Run `sum.py` to process the data and save results to `processed_train.csv`.


### Visualizations

- **`bow_true_ratio_histogram.png`**: A histogram showing the distribution of the **BoW/True** ratio, indicating how often BoW predictions align with the true similarity scores.

- **`tfidf_true_ratio_histogram.png`**: A histogram showing the distribution of the **TF-IDF/True** ratio, representing how often TF-IDF predictions align with the true similarity scores.

## How to Use

1. **Datasets**:
   - Use `stsb_train.csv` for training the models.
   - Use `stsb_validation.csv` for validation during training.
   - Use `stsb_test.csv` for evaluating final model performance.

2. **Model Evaluation**:
   - Results from the **BoW** and **TF-IDF** models, including predicted scores and performance metrics, are stored in `processed_train.csv`.
   - Use the **Pearson Correlation** and **Spearman's Rank Correlation** values to assess the models' performance in relation to the true similarity scores.

3. **Visualizations**:
   - Review `bow_true_ratio_histogram.png` and `tfidf_true_ratio_histogram.png` to understand how well the predicted scores from BoW and TF-IDF models correspond to the true similarity scores.

## Future Work

In future phases of this project, we will evaluate more advanced models, such as **Sentence-BERT (SBERT)**, to generate semantic embeddings and compare their performance with the baseline models (BoW and TF-IDF) on the same dataset.
