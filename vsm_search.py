#-------------------------------------------------------------------------
# AUTHOR: Porter Clevidence
# FILENAME: vsm_search.py
# SPECIFICATION: 
# You are provided with 100 documents, 5 queries, and binary relevance judgments for each query. 
# Using this data, complete the Python program (vsm_search.py) to build a Vector Space Model (VSM) 
# search engine and evaluate its performance.
#   Requirements: 
#    •	Represent documents and queries using TfidfVectorizer (remove stopwords only). 
#    •	For each query, score and rank all documents using cosine similarity. 
#    •	Using the relevance judgments, compute Average Precision (AP) for each query. 
#    •	Sort the queries in descending order of their AP values and report the ranking. 
# FOR: CS 5180- Assignment #3
# TIME SPENT: ~1 hour
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import csv

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# --> add your Python code here
INPUT_PATH = "docs.csv"
QUERY_PATH = 'queries.csv'
RELEVANCE_PATH = 'relevance_judgments.csv'

documents = []
with open(INPUT_PATH, 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
        if i > 0:  # skipping the header
           documents.append(row[1])
        
queries = []
with open(QUERY_PATH, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: 
            queries.append(row[1])

relevance = {}
with open(RELEVANCE_PATH, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        # row[0] is query id, row[1] is doc id, row[2] is relevance (R or N)
        if i > 0:
            rel = 1 if row[2] == 'R' else 0
            if row[0] not in relevance: relevance[row[0]] = {}
            relevance[row[0]][row[1]] = rel
            
# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# --> add your Python code here
tfidfvectorizer = TfidfVectorizer(
    stop_words='english'
)

training_v = tfidfvectorizer.fit_transform(documents)

tfidf_tokens = tfidfvectorizer.get_feature_names_out()
collection_df = pd.DataFrame(data=training_v.toarray(), index=[f'Doc{i}' for i in range(1, 101)], columns=tfidf_tokens)

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# --> add your Python code here
ap_scores = {}

for i, query in enumerate(queries):
    similarities = cosine_similarity(tfidfvectorizer.transform([query]), training_v)[0]
    
    ranked_docs = sorted(
        list(enumerate(similarities, start=1)),
        key=lambda x: x[1],
        reverse=True
    )
    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    rel = 0
    ap_sum = 0
    
    for rank, (doc_index, score) in enumerate(ranked_docs, start=1):
        if relevance.get(f'Q{i + 1}', {}).get(f'D{doc_index:03d}', 0) == 1:
            rel += 1
            ap_sum += rel / rank
   
    total_rel = sum(relevance.get(f'Q{i + 1}', {}).values())
    
    ap = ap_sum / total_rel if total_rel > 0 else 0
    
    # store the AP value for this query (use any data structure you prefer)
    ap_scores[f"Q{i + 1}"] = ap


# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
# --> add your Python code here
sorted_ap = sorted(ap_scores.items(), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
# --> add your Python code here
for query, ap in sorted_ap:
    print(f"{query}: {ap:.4f}")