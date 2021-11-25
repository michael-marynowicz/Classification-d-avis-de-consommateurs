from typing import Text
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.corpus import brown
from nltk.data import find
from sklearn.decomposition import PCA 
from sklearn.datasets import load_iris
from nltk.sentiment import SentimentIntensityAnalyzer
from math import *
import numpy as np 


def tokenise():
    file_content = open("./iphone8Anglais.txt").read()
    tokens = nltk.word_tokenize(file_content)
    tokensWithType = nltk.pos_tag(tokens)

    sentence =''
    for wt, type in tokensWithType:
        if (type.startswith('JJ')):
            sentence+=' '+wt
    return sentiment(sentence)


def trigrram(): #essais pour avoir le context
    file_content = open("./iphone8Anglais.txt").read()
    tokens = nltk.word_tokenize(file_content)
    output = list(nltk.trigrams(tokens))
    return output


def sentiment(sentence):
    sia = SentimentIntensityAnalyzer()
    return interprete(sia.polarity_scores(sentence)) 

def interprete(sentiment):
    
    negatif = sentiment['neg'] # les points negatif donne 1 etoile
    #neutre = sentiment['neu']*2.5  neutre je me dis qu'on les traitera en fonction du context non?
    positif = sentiment['pos']*5 # et positif on donne toutes les etoiles
    

    total=positif+negatif
    return ceil(total) # on retourne le total des etoiles 

print(tokenise())
