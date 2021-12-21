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
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import numpy as np
from numpy import array, empty
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csc_matrix

    # a faire :
        # utiliser word2vec

sia = SentimentIntensityAnalyzer()

def algo_with_svm():
    #RECUPERATION DU CSV
    data = pd.read_csv('reviews.csv')
    nom = data["Product Name"].head(30000)
    reviews = data['Reviews'].head(30000)
    """x_tokenized = [[w for w in sentence.split(" ") if w != ""] for sentence in reviews]
    model = gensim.models.Word2Vec(x_tokenized, min_count=1)"""
    
    
    
    #reviews = correction(reviews) utiliser la correction prend beaucoup de temps et n'augmente pas la precision
    rating = data['Rating'].head(30000)
    cat = categorie(rating)

#loading the model
    #model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True) 

#getting the vector for any word
    
    
    
    #Vectorizer les phrases
    vectorizer = CountVectorizer()
    reviews = vectorizer.fit_transform(reviews)
    #Split la liste entre la zone de training et de test 
    X_train , X_test , y_train, y_test = train_test_split(reviews, cat, test_size=0.70)
   
    #Declaration du modele
    clf_svc = svm.SVC(kernel='rbf', decision_function_shape='ovr')

    clf_svc1 = svm.SVC(kernel='rbf', decision_function_shape='ovr')
    clf_svc2 = svm.SVC(kernel='rbf', decision_function_shape='ovr')
    clf_svc3 = svm.SVC(kernel='rbf', decision_function_shape='ovr')

    #<class 'scipy.sparse.csr.csr_matrix'>

    X_train_very_good_and_neutre = []
    
    X_train_neutre_and_very_bad = []
    
    X_train_very_bad_and_very_good = []
    essais1=csc_matrix(X_train[0])
    essais2=csc_matrix(X_train[0])
    essais3=csc_matrix(X_train[0])

    y_train_very_good_and_neutre= []
    y_train_neutre_and_very_bad= []
    y_train_very_bad_and_very_good = []

    for i in range(len(y_train)):
        if (y_train[i]=="very good" or y_train[i]=="neutre"):
            if(essais1 is empty):
             
                essais1 = csc_matrix(X_train[i])
            else:
                essais1+=X_train[i]
            X_train_very_good_and_neutre.append(np.array(X_train[i]))
            
            y_train_very_good_and_neutre.append(y_train[i])
            
            #X_train_very_good_and_neutre.append(X_train[i])
        if (y_train[i]=="very bad" or y_train[i]=="neutre"):
          
            X_train_neutre_and_very_bad.append(np.array(X_train[i]))
            if(essais2 is empty):
                essais2 = csc_matrix(X_train[i])
            else:
                essais2+=X_train[i]
            
            y_train_neutre_and_very_bad.append(y_train[i])
            #X_train_neutre_and_very_bad.append(X_train[i])
        if(y_train[i]=="very bad" or y_train[i]=="very good"):
            
            m = X_train_very_bad_and_very_good.append(np.array(X_train[i]))
            if(essais3 is empty):
                essais3 = csc_matrix(X_train[i])
            else:
                essais3+=X_train[i]
            
            y_train_very_bad_and_very_good.append(y_train[i])



            #X_train_very_bad_and_very_good.append(X_train[i])
    print(type(essais1),type(y_train),type(y_train_very_good_and_neutre))
    print(essais1.shape,len(y_train_very_good_and_neutre))
    clf_svc1.fit(essais1, y_train_very_good_and_neutre) # j'ai essayé de convertir X_train_very_good_and_neutre mais ca bug 
    clf_svc2.fit(X_train_neutre_and_very_bad, y_train_neutre_and_very_bad) # bug a cause de X_train_neutre_and_very_bad qui n'est pas une csr_matrix
    clf_svc3.fit(X_train_very_bad_and_very_good, y_train_very_bad_and_very_good)# bug 
    
    #Le modèle prédit sur la zone de test 
    result1 = clf_svc1.predict(X_test) 
    result2 = clf_svc2.predict(X_test) 
    result3 = clf_svc3.predict(X_test) 

    #statistiques
    print("very good and neutre = ",classification_report(y_test, result1))
    print("neutre and very bad = ",classification_report(y_test, result2))
    print("very good and very bad = ",classification_report(y_test, result3))
    

    


    #On compare le resultat aux notes
    print("very good and neutre = ",confusion_matrix(result1, y_test))
    print("neutre and very bad = ",confusion_matrix(result2, y_test))
    print("very good and very bad = ",confusion_matrix(result3, y_test))
    #print(reviews)

    
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


