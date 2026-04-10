## Assignment
You are provided with 100 documents, 5 queries, and binary relevance judgments for each query. Using this data, complete the Python program (vsm_search.py) to build a Vector Space Model (VSM) search engine and evaluate its performance.
### Requirements: 
- Represent documents and queries using TfidfVectorizer (remove stopwords only). 
- For each query, score and rank all documents using cosine similarity. 
- Using the relevance judgments, compute Average Precision (AP) for each query. 
- Sort the queries in descending order of their AP values and report the ranking. 