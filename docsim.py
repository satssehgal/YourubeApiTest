from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 

set(stopwords.words('english'))

def YTCompare(searchterms, title):
	stop_words = stopwords.words('english')
	text1 = searchterms
	text2 = title
	processed_doc1=' '.join([word for word in text1.split() if word not in stop_words])
	processed_doc2=' '.join([word2 for word2 in text2.split() if word2 not in stop_words])
	corpus = [processed_doc1, processed_doc2]
	vectorizer = TfidfVectorizer()
	tfidf = vectorizer.fit_transform(corpus)
	similarity_matrix = cosine_similarity(tfidf)[0,1]
	return similarity_matrix

