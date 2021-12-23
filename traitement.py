from re import M
import nltk
import gensim
from textblob import TextBlob
nltk.downloader.download('vader_lexicon')
nltk.download('word2vec_sample')
from nltk.data import find
from nltk.sentiment import SentimentIntensityAnalyzer
from math import *
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from numpy import array, empty
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score
from pandas import *
from sklearn import *


    # a faire :
        # utiliser word2vec

sia = SentimentIntensityAnalyzer()

def algo_with_svm():
    #RECUPERATION DU CSV
    data = pd.read_csv('reviews.csv')
    nom = data["Product Name"].head(5000)
    reviews = data['Reviews'].head(5000)

    #reviews = correction(reviews) utiliser la correction prend beaucoup de temps et n'augmente pas la precision
    rating = data['Rating'].head(5000)
    cat = categorie(rating)
    
    #Vectorizer les phrases
    vectorizer = CountVectorizer()
    reviews = vectorizer.fit_transform(reviews)


    #Split la liste entre la zone de training et de test 
    X_train , X_test , y_train, y_test = train_test_split(reviews, cat, test_size=0.30)
   
    #Declaration du modele

    clf_svc1 = svm.SVC(kernel='rbf', decision_function_shape='ovr')
    clf_svc2 = svm.SVC(kernel='rbf', decision_function_shape='ovr')
    clf_svc3 = svm.SVC(kernel='rbf', decision_function_shape='ovr')

    #<class 'scipy.sparse.csr.csr_matrix'>

    X_train_very_good_and_neutre = []
    X_test_very_good_and_neutre = []

    X_train_neutre_and_very_bad = []
    X_test_neutre_and_very_bad = []

    X_train_very_bad_and_very_good = []
    X_test_very_bad_and_very_good = []

    y_train_very_good_and_neutre= []
    y_test_very_good_and_neutre= []

    y_train_neutre_and_very_bad= []
    y_test_neutre_and_very_bad= []

    y_train_very_bad_and_very_good = []
    y_test_very_bad_and_very_good = []

    for i in range(len(y_test)):
        if (y_test[i]=="very good" or y_test[i]=="neutre"):
            X_test_very_good_and_neutre.append(X_test[i].toarray()[0])
            
            y_test_very_good_and_neutre.append(y_test[i])

        if (y_test[i]=="very bad" or y_test[i]=="neutre"):
          
            
            X_test_neutre_and_very_bad.append(X_test[i].toarray()[0])
            
            y_test_neutre_and_very_bad.append(y_test[i])

        if(y_test[i]=="very bad" or y_test[i]=="very good"):           
            
            X_test_very_bad_and_very_good.append(X_test[i].toarray()[0])
            
            y_test_very_bad_and_very_good.append(y_test[i])


    for i in range(len(y_train)):

        if (y_train[i]=="very good" or y_train[i]=="neutre"):

            X_train_very_good_and_neutre.append(X_train[i].toarray()[0])
            
            y_train_very_good_and_neutre.append(y_train[i])

        if (y_train[i]=="very bad" or y_train[i]=="neutre"):
          
            X_train_neutre_and_very_bad.append(X_train[i].toarray()[0])
            
            y_train_neutre_and_very_bad.append(y_train[i])
       
        if(y_train[i]=="very bad" or y_train[i]=="very good"):
            
            X_train_very_bad_and_very_good.append(X_train[i].toarray()[0])
            
            y_train_very_bad_and_very_good.append(y_train[i])
    
    
    X_train_very_good_and_neutre = sparse.csc_matrix(X_train_very_good_and_neutre)
    X_train_neutre_and_very_bad = sparse.csc_matrix(X_train_neutre_and_very_bad)
    X_train_very_bad_and_very_good = sparse.csc_matrix(X_train_very_bad_and_very_good)

    X_test_very_good_and_neutre = sparse.csc_matrix(X_test_very_good_and_neutre)
    X_test_neutre_and_very_bad = sparse.csc_matrix(X_test_neutre_and_very_bad)
    X_test_very_bad_and_very_good = sparse.csc_matrix(X_test_very_bad_and_very_good)
    
    clf_svc1.fit(X_train_very_good_and_neutre, y_train_very_good_and_neutre)
    clf_svc2.fit(X_train_neutre_and_very_bad, y_train_neutre_and_very_bad) 
    clf_svc3.fit(X_train_very_bad_and_very_good, y_train_very_bad_and_very_good)

    """display = PrecisionRecallDisplay.from_estimator(
    clf_svc1, X_test_very_good_and_neutre, y_test_very_good_and_neutre, name="LinearSVC")
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    input()"""
    
    #Le modèle prédit sur la zone de test 
    result1 = clf_svc1.predict(X_test_very_good_and_neutre) 
    result2 = clf_svc2.predict(X_test_neutre_and_very_bad) 
    result3 = clf_svc3.predict(X_test_very_bad_and_very_good) 
   
    #statistiques
    print("very good and neutre = \n",classification_report(y_test_very_good_and_neutre, result1))
    print("neutre and very bad = \n",classification_report(y_test_neutre_and_very_bad, result2))
    print("very good and very bad = \n",classification_report(y_test_very_bad_and_very_good, result3))
    
    #On compare le resultat aux notes
    print("very good and neutre = \n",confusion_matrix(result1, y_test_very_good_and_neutre))
    print("neutre and very bad = \n",confusion_matrix(result2, y_test_neutre_and_very_bad))
    print("very good and very bad = \n",confusion_matrix(result3, y_test_very_bad_and_very_good))

    y_final = []
    
    

    """commentaire = "it's a really good phone"
    c_l = list()
    c_l.append(commentaire)
    c_l = vectorizer.fit_transform(c_l)
    c_l_matrix = sparse.csc_matrix(c_l)
    print(np.shape(X_train))
    print(np.shape(c_l_matrix))
    print(X_train[0])
    print(X_train[0].todense())
    print(c_l_matrix[0])

    clf_svc1.predict([[0,3,4]]) """












    cat_commentaire1 = clf_svc1.predict(X_test)
    cat_commentaire2 =clf_svc2.predict(X_test)
    cat_commentaire3 =clf_svc3.predict(X_test)
    print(type([y_test[0]]))
    for i in range(len(y_test)):
        o1=0
        o2=0
        o3 = 0
        if (y_test[i]=="very good" or y_test[i]=="neutre"):
            print(y_test[i],[cat_commentaire1[i]])
            o1 = precision_score(y_test[i],cat_commentaire1[i],pos_label=['neutre', 'very good'])
        if (y_test[i]=="very bad" or y_test[i]=="neutre"):
            o2 = precision_score(y_test[i],cat_commentaire2[i],pos_label=['neutre', 'very bad'])
        if (y_test[i]=="very good" or y_test[i]=="very bad"):
            o3 = precision_score(y_test[i],cat_commentaire3[i],pos_label=['very bad', 'very good'])

        if (max(o1,o2,o3)==o1):
        
            y_final.append(cat_commentaire1[i])

        elif (max(o1,o2,o3)==o2):
     
            y_final.append(cat_commentaire2[i])

        else:

            y_final.append(cat_commentaire3[i])

    print("y final = \n",confusion_matrix(y_final, y_test))
    
    
    

 
    return 0
    


def correction(sentences):
    sentences_corrige = []
    for i in range (len(sentences)):
        corrige = TextBlob(sentences[i])
        sentences_corrige.append(str(corrige.correct()))
    return sentences_corrige

def categorie(tab):
    cat = []
    for i in range(len(tab)):
        if (tab[i]==1):
            cat.append("very bad")
        elif (tab[i]>1 and tab[i]<5):
            cat.append("neutre")
        else: 
            cat.append("very good")
    verybad = 0
    neutre=0
    verygood=0
    for i in range(len(cat)):
        if (cat[i]=="very bad"):
            verybad+=1
        elif (cat[i]=="neutre"):
            neutre+=1
        elif (cat[i]=="very good"):
            verygood+=1
            
    print("very bad = ",verybad,"neutre = ",neutre,"very good = ",verygood)
    return cat
print(algo_with_svm())

########################################### a garder pour citer dans le rapport ##################################################################
"""for i in range(len(reviews)):# au lieu de prendre la review en entier on prend seulement les adjectifs de la reviews
        tokens = nltk.word_tokenize(reviews[i])
        tokensWithType = nltk.pos_tag(tokens)
        sentence =''
        for wt, type in tokensWithType:
            if (type.startswith('JJ')):
                sentence+=' '+wt
        reviews[i]=sentence"""

def essais():
    file_content = pd.read_csv('reviews.csv')
    
    tokens = nltk.word_tokenize("gooddd")
    tokensWithType = nltk.pos_tag(tokens)
    sentence =''
    for wt, type in tokensWithType:
        if (type.startswith('JJ')):
            sentence+=' '+wt
    neutre = extract_neutre(sentence)

    context_list,sentence_without_neutre = context(file_content,neutre) 

    note_neutre = evalue_with_context(context_list)
    return get_note(sentence_without_neutre,note_neutre)
def determine_polarity(text):
    blob = TextBlob(text)
    acc=0
    result=0
    for sentence in blob.sentences:
        result+=(sentence.sentiment.polarity+1)*2.5
        acc+=1
    return result/acc


    
def evalue_with_context(all_neutre_list):
    note=0
    acc=0
    for i in range(len(all_neutre_list)):
        note +=determine_polarity(all_neutre_list[i])
        acc+=1
    return note/acc

def context(file_content,neutres): #essais pour avoir le context
    context_of_neutre_word = []
    file_content_list = list(file_content.split(" "))
    for i in range(len(file_content_list)):
        if (file_content_list[i] in neutres and (i>0 and i<len(file_content_list)-1)):
            context = file_content_list[i-1]+' '+file_content_list[i]+' '+file_content_list[i+1]
            context_of_neutre_word.append(context)
            file_content=file_content.replace(context,"")
            
    return context_of_neutre_word,file_content

def extract_neutre(sentence):

    neutre = ''
    sentence_list = list(sentence.split(" "))
    for i in range(len(sentence_list)):
        if (sia.polarity_scores(sentence_list[i])['neu']==1):
            neutre+=sentence_list[i]+' '
    return neutre

def get_sentiment(sentence):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(sentence)

def get_synonyme(word):
    word2vec_sample =str(find('models/word2vec_sample/pruned.word2vec.txt'))
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    return model.most_similar(positive=[word], topn = 1)

def get_note(content,note_neutre):
    print(determine_polarity(content))
    return (determine_polarity(content)+note_neutre)/2 # on retourne le total des etoiles 


