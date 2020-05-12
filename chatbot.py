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
    
InputGreetings=('Namaste','hi','hello','heya','howdy','greetings','hey','sup','hey there')
ResponseGreetings=('hi!!','heya!','howdy:)','Namaste','hey there:D','glad you are talking to me!',)
def greeting(sentence):
        
    for word in sentence.split():
        if word.lower() in InputGreetings:
            return random.choice(ResponseGreetings)
#the above code returns random response greetings from the above in response to the user.

#STEP 4 : VECTORIZER
            
from sklearn.feature_extraction.text import TfidfVectorizer
#this library is used to convert words into vectors/arrays
from sklearn.metrics.pairwise import cosine_similarity
#the cosine similarity library helps find similarities between the user questions and our data
def response(user_response):
    chatbot_response=''
    sentenceTokens.append(user_response)
    TfidfVec=TfidfVectorizer(tokenizer=Lemmer_Normalize,stop_words='english')
    tfidf=TfidfVec.fit_transform(sentenceTokens)
    #learning from our sentence tokens and converting it into vectors
    vals=cosine_similarity(tfidf[-1],tfidf)
    #tries to find similarities between text tokens and user questions
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    #converting values to row/column matrix
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):# no match found
        chatbot_response=chatbot_response+"I'm sorry, I'm not sure if I understood what you just said!."
        return chatbot_response
    else:
        chatbot_response=chatbot_response+sent_tokens[idx]
        return chatbot_response





