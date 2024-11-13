import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import numpy as np
# 1. 加载 stsb_validation.csv 数据
validation_df = pd.read_csv('stsb_train.csv')

# 提取句子对和真实相似度分数 (score 列)
sentence_pairs = validation_df[['sentence1', 'sentence2']].values.tolist()
true_scores = validation_df['score'].tolist()

# 2. 提取所有句子用于向量化
all_sentences = [sentence for pair in sentence_pairs for sentence in pair]

# 3. Bag-of-Words (BoW) 基线模型
vectorizer = CountVectorizer()
sentence_vectors_bow = vectorizer.fit_transform(all_sentences).toarray()

# 计算 BoW 余弦相似度
predicted_scores_bow = []
for i in range(0, len(sentence_vectors_bow), 2):
    cos_sim = cosine_similarity([sentence_vectors_bow[i]], [sentence_vectors_bow[i+1]])[0][0]
    predicted_scores_bow.append(cos_sim)

# 4. TF-IDF 基线模型
tfidf_vectorizer = TfidfVectorizer()
sentence_vectors_tfidf = tfidf_vectorizer.fit_transform(all_sentences).toarray()

# 计算 TF-IDF 余弦相似度
predicted_scores_tfidf = []
for i in range(0, len(sentence_vectors_tfidf), 2):
    cos_sim = cosine_similarity([sentence_vectors_tfidf[i]], [sentence_vectors_tfidf[i+1]])[0][0]
    predicted_scores_tfidf.append(cos_sim)

# 5. 评估 BoW 模型性能
pearson_corr_bow, _ = pearsonr(predicted_scores_bow, true_scores)
spearman_corr_bow, _ = spearmanr(predicted_scores_bow, true_scores)

# 6. 评估 TF-IDF 模型性能
pearson_corr_tfidf, _ = pearsonr(predicted_scores_tfidf, true_scores)
spearman_corr_tfidf, _ = spearmanr(predicted_scores_tfidf, true_scores)

# 7. 打印结果
print(f"BoW Cosine Similarity Scores: {predicted_scores_bow[:5]}")  # 打印前5个相似度分数
print(f"BoW Pearson Correlation: {pearson_corr_bow}")
print(f"BoW Spearman Rank Correlation: {spearman_corr_bow}")

print(f"TF-IDF Cosine Similarity Scores: {predicted_scores_tfidf[:5]}")  # 打印前5个相似度分数
print(f"TF-IDF Pearson Correlation: {pearson_corr_tfidf}")
print(f"TF-IDF Spearman Rank Correlation: {spearman_corr_tfidf}")
#能不能把这些结果添加到原表格成为新表格processed_train.csx
#原本包含sentence1 sentence2 score
#现在包含sentence1 sentence2 score BoW Cosine Similarity Scores TF-IDF Cosine Similarity Scores
#  BoW Pearson Correlation BoW Spearman Rank Correlation TF-IDF Pearson Correlation TF-IDF Spearman Rank Correlation
# 8. 将 BoW 和 TF-IDF 余弦相似度分数添加到原始 DataFrame 中
validation_df['BoW score'] = predicted_scores_bow
validation_df['TF-IDF score'] = predicted_scores_tfidf

#9. 将 Pearson 和 Spearman 相关系数添加到 DataFrame 中
#因为所有行这个相关一样 只要第二行记录这个pearson spearman值就可以
'''validation_df['BoW Pearson'] = pearson_corr_bow
validation_df['BoW Spearman'] = spearman_corr_bow
validation_df['TF-IDF Pearson'] = pearson_corr_tfidf
validation_df['TF-IDF Spearman'] = spearman_corr_tfidf'''
#将 Pearson 和 Spearman 相关系数仅添加到第二行
validation_df['BoW Pearson Correlation'] = None
validation_df['BoW Spearman Rank Correlation'] = None
validation_df['TF-IDF Pearson Correlation'] = None
validation_df['TF-IDF Spearman Rank Correlation'] = None

validation_df.at[0, 'BoW Pearson Correlation'] = pearson_corr_bow
validation_df.at[0, 'BoW Spearman Rank Correlation'] = spearman_corr_bow
validation_df.at[0, 'TF-IDF Pearson Correlation'] = pearson_corr_tfidf
validation_df.at[0, 'TF-IDF Spearman Rank Correlation'] = spearman_corr_tfidf

#  计算 BoW/True 和 TF-IDF/True 的比例
validation_df['BoW/True'] = validation_df['BoW score'] / validation_df['score']
validation_df['TF-IDF/True'] = validation_df['TF-IDF score'] / validation_df['score']

# 处理无穷大的情况（如果真实得分为 0）
validation_df.replace([np.inf, -np.inf], np.nan, inplace=True)
validation_df.fillna(0, inplace=True)
# 10. 保存处理后的数据到 CSV 文件
processed_file_path = 'processed_train.csv'
validation_df.to_csv(processed_file_path, index=False)


#打印新表前五行
print(validation_df.head())

# 设置区间范围 (bins) 用于统计 BoW/True 和 TF-IDF/True 的频率
bins = np.linspace(0, 5, 50)  # 区间范围从 0 到 5，划分为 50 个区间

# 1. 绘制 BoW/True 的直方图
plt.figure(figsize=(10, 6))
plt.hist(validation_df['BoW/True'], bins=bins, color='blue', alpha=0.7, label='BoW/True Ratio')
plt.title('Distribution of BoW/True Ratios')
plt.xlabel('BoW Cosine Similarity / True Score Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()

# 保存和显示 BoW/True 的图表
plt.tight_layout()
plt.savefig('bow_true_ratio_histogram.png')
plt.show()

# 2. 绘制 TF-IDF/True 的直方图
plt.figure(figsize=(10, 6))
plt.hist(validation_df['TF-IDF/True'], bins=bins, color='green', alpha=0.7, label='TF-IDF/True Ratio')
plt.title('Distribution of TF-IDF/True Ratios')
plt.xlabel('TF-IDF Cosine Similarity / True Score Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()

# 保存和显示 TF-IDF/True 的图表
plt.tight_layout()
plt.savefig('tfidf_true_ratio_histogram.png')
plt.show()
