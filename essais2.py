import nltk
import gensim
from textblob import TextBlob
nltk.downloader.download('vader_lexicon')
nltk.download('word2vec_sample')
from nltk.data import find
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd


sia = SentimentIntensityAnalyzer()

########################################### a garder pour citer dans le rapport ##################################################################


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

