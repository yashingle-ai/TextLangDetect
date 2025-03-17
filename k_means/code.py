import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from sklearn.metrics import adjusted_rand_score

# Load Language Files
data_folder = r"C:\\Users\\yashi\\OneDrive\\Desktop\\fasttext\\dataset"  # Change to your folder path
language_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".txt")])

sentences = []
language_labels = []
language_names = {}

for idx, file in enumerate(language_files):
    file_path = os.path.join(data_folder, file)
    language_names[idx] = file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        sentences.extend([line.strip() for line in lines])
        language_labels.extend([idx] * len(lines))

# Convert Sentences into Numerical Vectors using Character N-Gram TF-IDF
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=30000)  # Reduce features for memory
tfidf_matrix = vectorizer.fit_transform(sentences)  # Keep sparse matrix

#  Reduce Dimensionality using **Truncated SVD (Better than PCA for Sparse Data)**
svd = TruncatedSVD(n_components=500)  # Reduce to 100 dimensions
tfidf_reduced = svd.fit_transform(tfidf_matrix)

#  Apply MiniBatchKMeans Clustering
num_clusters = 22  # 22 languages
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=5000, n_init=10)
kmeans.fit(tfidf_reduced)

#  Assign clusters
clusters = kmeans.labels_

# Generate Cluster Report
language_cluster_map = {i: [] for i in range(num_clusters)}

for i, cluster in enumerate(clusters):
    language_cluster_map[cluster].append(language_labels[i])

print("\n--- Clustering Report ---\n")
for cluster, lang_indices in language_cluster_map.items():
    most_common_lang = Counter(lang_indices).most_common(1)
    if most_common_lang:
        most_common_lang_id, count = most_common_lang[0]
        accuracy = count / len(lang_indices) * 100
        print(f"Cluster {cluster}:")
        print(f"  - Most common language: {language_names[most_common_lang_id]}")
        print(f"  - Total sentences in cluster: {len(lang_indices)}")
        print(f"  - Accuracy within cluster: {accuracy:.2f}%\n")

#  Evaluate Overall Performance
ari_score = adjusted_rand_score(language_labels, clusters)
print(f"Overall Clustering Quality (Adjusted Rand Index): {ari_score:.4f}")

#  Print Sample Sentences with Assigned Clusters
print("\n--- Sample Sentences with Assigned Clusters ---\n")
for i in range(10):  # Print first 10 sentences
    print(f"Sentence: {sentences[i]}")
    print(f"Actual Language: {language_names[language_labels[i]]}")
    print(f"Assigned Cluster: {clusters[i]}\n")
