This repository contains files and resources related to our Semantic Similarity Detection project, where we evaluate traditional models like Bag-of-Words (BoW) and TF-IDF, using the STS Benchmark (STSb) dataset. The project includes both the datasets and the analysis results, as well as performance evaluations based on metrics like Cosine Similarity, Pearson Correlation, and Spearman's Rank Correlation.

Files Overview
1. stsb_train.csv, stsb_test.csv, stsb_validation.csv
These three files represent the STS Benchmark (STSb) dataset, which is split into training, testing, and validation sets. Each file contains pairs of sentences and their corresponding similarity score, normalized between 0 and 1.

stsb_train.csv: The training set, used to train our models.
stsb_test.csv: The test set, used to evaluate model performance after training.
stsb_validation.csv: The validation set, used to tune and validate the models during training.
Each CSV file contains the following columns:

sentence1: The first sentence in the pair.
sentence2: The second sentence in the pair.
score: The human-annotated similarity score (0 to 1).
2. processed_train.csv
This file contains the processed data after running our BoW and TF-IDF models on the training set (stsb_train.csv). In addition to the original columns (sentence1, sentence2, and score), this file includes the following additional columns:

BoW score: The predicted similarity score based on the Bag-of-Words (BoW) model.
TF-IDF score: The predicted similarity score based on the TF-IDF model.
BoW Pearson Correlation: The Pearson correlation value for the BoW model (only populated in the first row).
BoW Spearman Rank Correlation: The Spearman rank correlation value for the BoW model (only populated in the first row).
TF-IDF Pearson Correlation: The Pearson correlation value for the TF-IDF model (only populated in the first row).
TF-IDF Spearman Rank Correlation: The Spearman rank correlation value for the TF-IDF model (only populated in the first row).
BoW/True: The ratio of BoW predicted score to the true similarity score.
TF-IDF/True: The ratio of TF-IDF predicted score to the true similarity score.
This file is useful for analyzing how well each model aligns with the actual similarity scores.

3. bow_true_ratio_histogram.png
This is a histogram that visualizes the distribution of the BoW/True ratio, showing how the BoW predicted similarity scores compare to the actual similarity scores. The x-axis represents the ratio of predicted to true similarity, while the y-axis represents the frequency of each ratio value.

4. tfidf_true_ratio_histogram.png
This is a histogram that visualizes the distribution of the TF-IDF/True ratio, showing how the TF-IDF predicted similarity scores compare to the actual similarity scores. The x-axis represents the ratio of predicted to true similarity, while the y-axis represents the frequency of each ratio value.

Usage Instructions
Datasets: Use the stsb_train.csv for model training, stsb_validation.csv for validation, and stsb_test.csv for final model evaluation.
Model Performance: Analyze the performance of BoW and TF-IDF models by checking the processed_train.csv file, which contains both the predicted scores and the comparison metrics like Pearson and Spearman correlations.
Visualizations: Review the bow_true_ratio_histogram.png and tfidf_true_ratio_histogram.png to visually compare how well the BoW and TF-IDF models predict similarity scores relative to the actual scores.
Future Work
In future work, we plan to extend this analysis by evaluating Sentence-BERT (SBERT), a transformer-based model, to compare its performance with these traditional methods on the same dataset.

