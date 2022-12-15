# Import the necessary libraries
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Set the number of clusters
num_clusters = 5

# Read the keywords from the text file
with open('keywords.txt', 'r') as f:
  keywords = f.readlines()

# Create a CountVectorizer object to convert the keywords into a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(keywords)

# Use K-Means clustering to automatically categorize the keywords
km = KMeans(n_clusters=num_clusters)
km.fit(X)

# Write the categorization results to a CSV file
with open('keywords_categorized.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerows(zip(keywords, km.labels_))
