# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:09:56 202

@author: harnoor
"""
import nltk
import io
#library that deals with opening and reading a file

import numpy as np
#library to perform mathematical operations

import random
#used for random selection of words from text

import string 
#this library will be used to process standard python strings
import warnings
warnings.filterwarnings('ignore')
#this would ignore any earning that is incurred while running our application

#STEP 2 :

file=open('D:\chatbot\globalWarming.txt','r',errors='ignore') 
#setting target to our file
data=file.read()
#allows to read data from our target file
data=data.lower()
#converts data to lower
nltk.download('punkt')
#used for sentence tokenization i.e it converts a string to a list of words
nltk.download('wordnet')
#lexical database for english language, nltk refers to it during run
sentenceTokens=nltk.sent_tokenize(data)
#converting text into sentences: creating sentence tokens
wordTokens=nltk.word_tokenize(data)
#converting the sentences to words: creating word tokens
sentenceTokens[:50]
wordTokens[:50]


#STEP 3 :PREPROCESSING
lemmers=nltk.stem.WordNetLemmatizer()
#it ensures that the root word to which our word has been converted is a valid word
def Lemmer_Tokens(tokens):
    return [lemmers.lemmatize(token) for token in tokens]
removePunctuation=dict((ord(punct),None) for punct in string.punctuation)
#removes the punctuation marks in data
def Lemmer_Normalize(text):
    return Lemmer_Tokens(nltk.word_tokenize(text.lower().translate(removePunctuation)))
#the above function takes up user queries, converts into lower case and removes punctuation marks if any

#STEP 4 : SETTING UP GREETINGS
    








