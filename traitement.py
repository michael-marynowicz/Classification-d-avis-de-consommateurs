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
    return interprete(sia.polarity_scores("A little more compact and lightweight than its predecessor, the iPhone 12 nevertheless carries many improvements, such as its OLED screen which, failing to offer the best contrast, shines with its perfectly calibrated colors or almost. The latest iPhone also gains access to the 5G network and benefits from an even more impressive A14 Bionic chip than the A13 of the previous generation. However, there is still a lack of improvement in the function that would probably have needed it the most, namely the photo. While the two 12-megapixel modules of the iPhone 12 are far from bad, they remain behind what the direct competition offers, in terms of versatility at least, and often also in image quality. We also regret the lack of progress in autonomy and, above all, that Apple decided to remove the AC adapter from the cabinet while the transition to USB-C is still far from complete. Newcomers to the Apple ecosystem may quickly become disillusioned with the Lightning. It will take for the others come from a model older than the iPhone 11 to find an interest other than the support of the 5G to this iPhone 12, certainly very good, but ultimately only strengthening the assets.")) 

def interprete(sentiment):
    
    negatif = sentiment['neg'] # les points negatif donne 1 etoile
    #neutre = sentiment['neu']*2.5  neutre je me dis qu'on les traitera en fonction du context non?
    positif = sentiment['pos']*5 # et positif on donne toutes les etoiles
    

    total=positif+negatif
    return ceil(total) # on retourne le total des etoiles 

print(tokenise())
