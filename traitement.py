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
    print(y_test[0])
    #Entrainement du modele
    clf_svc.fit(X_train, y_train)

    print(X_test)
    
    #Le modèle prédit sur la zone de test 
    result = clf_svc.predict(X_test)
    #statistiques
    print(classification_report(y_test, result))
    

    


    #On compare le resultat aux notes
    print(confusion_matrix(result, y_test))
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
    good=0
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


