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
# TIME SPENT: 
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# --> add your Python code here


# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# --> add your Python code here


    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------

    # store the AP value for this query (use any data structure you prefer)


# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
# --> add your Python code here