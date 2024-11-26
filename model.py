#Candidate Name: Suvankar Biswas
#Colege: Jadavpur University

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

#Loading Data
materials = pd.read_csv('materials.csv')
test_pairs = pd.read_csv('test_pairs.csv')

#Preprocessing data

def preprocess_text(text):
    text = text.lower() #lowercase conversion
    # Remove special characters, numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

materials['Processed_Description'] = materials['Material_Description'].apply(preprocess_text)

# #printing some entries of cleaned data
# print("\nCleaned Material Descriptions:")
# print(materials[['Material_Description', 'Processed_Description']].head())

#Tf-IDF vectors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(materials['Processed_Description'])

def get_similarity_scores(test_pairs, tfidf_matrix, materials):
    similarities = []
    for _, row in test_pairs.iterrows():
        idx1 = materials[materials['ID'] == row['ID_1']].index[0]
        idx2 = materials[materials['ID'] == row['ID_2']].index[0]
        score = cosine_similarity(tfidf_matrix[idx1], tfidf_matrix[idx2])
        rounded_score = round(score[0][0], 2)
        similarities.append(rounded_score)
        
    return similarities


#Similarity Calculation
test_pairs['Similarity_Score'] = get_similarity_scores(test_pairs, tfidf_matrix, materials)

#Display similarity scores
print("\nTest Pairs with Similarity Scores:")
print(test_pairs.head())

# Saving smilarity scores to submission.csv
test_pairs[['ID_1', 'ID_2', 'Similarity_Score']].to_csv('submission.csv', index=False)
print("\nSimilarity scores saved to submission.csv!")