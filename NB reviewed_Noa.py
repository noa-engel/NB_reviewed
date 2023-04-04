#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:16:05 2023

@author: noaengel
"""

#1 - BUSINESS UNDERSTANDING

#The code is using a dataset of news articles labeled as either real or fake,
# and it is training a machine learning model (a Naive Bayes classifier) to classify new articles as 
# either real or fake based on the words used in the article text. The goal of this code is to help 
# identify fake news articles, which can have negative impacts on society by spreading misinformation
# and undermining public trust in the media.

# 2 - DATA UNDERSTANDING

# Below are the imported packages and their libraries
import pandas as pd
from pandas.api.types import CategoricalDtype
from wordcloud import WordCloud
from matplotlib import pyplot as plt

# Importing dataset
fake_news = pd.read_csv('https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/NB-fakenews.csv')
fake_news.head()


# Creating a categorical data type for the 'label' column in the 'fake_news' dataset.
# The categories are set to be 0 and 1, which are the labels used to indicate whether an article is real or fake.

catID = CategoricalDtype(categories = [0,1], ordered = False) #Noa: I used numbers instead of real and fake because otherwise you get NAN for label

fake_news.label = fake_news.label.astype(catID)
fake_news.label

fake_news.label.value_counts()

fake_news.label.value_counts(normalize = True)

# These lines create strings of all the titles of news articles in the 'fake_news' dataset that are labeled as either 0 or 1, which respectively defines whether they are fake or real news.

untrue_text = " ".join(str(text) for text in fake_news[fake_news['label'] == 0]['title'].values)
real_text = " ".join(str(text) for text in fake_news[fake_news['label'] == 1]['title'].values) #Noa: Hossein mentioned lines as in multiple lines, which was a good clue. And this one should not be missing
#besides, you need conditions for both numbers

# Creating WordCloud objects that visualize the most common words in either 'untrue_text' or 'real_text' strings.
wc_untrue = WordCloud(background_color='white', colormap='Blues').generate(untrue_text)
wc_real = WordCloud(background_color='white', colormap='Reds').generate(real_text)

# Generating and displaying two word clouds side-by-side for the fake news and real news titles.
# Function plt.subplots() creates a figure object and two subplots, and returns them in a tuple.
# Sets the title of the figure to "Wordclouds for untrue and real".
# The two imshow() functions that follow, they display the generated word clouds in the corresponding subplots.
fig, (wc1, wc2) = plt.subplots(1,2)
fig.suptitle('Wordclouds for untrue and real') 
wc1.imshow(wc_untrue)
wc2.imshow(wc_real)
plt.show() #Visualising

#3 - DATA PREPARATION

# Vectors 
# Responsible for checking and handling missing data in the 'text' column of the fake_news dataframe.
print(fake_news['text'].isna().sum())
fake_news.dropna(subset=['text'], inplace=True) #Noa: to make sure that all NA values remaining are dropped

# Importing extra packages from their libraries. I added them here, because they are all part of sklearn.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score #Noa: I missed a few packages

# Using the TfidfVectorizer class from the scikit-learn library, I convert the text data into a numerical form that can be used for machine learning algorithms.
# (max_features=1000) means that only the top 1000 most frequently occurring words in the dataset will be used as features.
vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(fake_news.text) #Noa: text instead of label, otherwise it won't work
wordsfake_news = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())
wordsfake_news.head()
# The resulting DataFrame is stored in the variable wordsfake_news, which will be used for the subsequent machine learning analysis.

# Splitting the dataset into training and testing sets.
xTrain, xTest, yTrain, yTest = train_test_split(wordsfake_news, fake_news.label)


#4 - DATA MODELING 

# Testing the data
# These lines of code are related to the classification task of the dataset. Specifically, they are creating an instance of the Multinomial Naive Bayes model.
# The Multinomial Naive Bayes variant is specifically designed for text classification tasks where the features (words in the text) are discrete counts,
# such as the number of times a word appears in a document.
bayes = MultinomialNB()
bayes.fit(xTrain, yTrain)

yPred = bayes.predict(xTest)
yTrue = yTest

# This part of the code is computing the accuracy score of the Naive Bayes model. 
# In particular, the accuracy_score function compares the predicted labels (yPred) with the true labels (yTrue) for the test data.
accuracyScore = accuracy_score(yTrue, yPred) #Noa:here I saw that the dot needed to be an underscore
print(f'Accuracy: {accuracyScore}') #Noa: the accuracy of the model is 87.5%

#5 - DATA EVALUATION 

# Then, we use confusion matrix, which is a table used to evaluate the performance of a classification model by summarizing
# the number of correct and incorrect predictions made by the model.
matrix = confusion_matrix(yTrue, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot() #Noa: Made the confusion matrix visible

