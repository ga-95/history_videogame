import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import stem
it_stem=nltk.stem.SnowballStemmer('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#from google_trans_new import google_translator
import re
import numpy as np
import pandas as pd
from pprint import pprint
import numbers
#from langdetect import detect
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import glob
import sys
#import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
# spacy for lemmatization
import spacy
#import mallet

# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline

import importlib

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend([ 'u', 'https', 'www', 'youtube',  'com', 'removed', 'http',  'wikipedia','9z3vqo', 'en',
                    'org', 'wiki', 'rep', 'like', 'wikia', 'youtub', 'yldaukrnl2q', 'r',
                    '[deleted]', 'deleted','delete', 'delet', '™','TM', "html", 'oh'])
#'would', 'want', 'could', 'go', 'get',

#tokenizing
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

diz_lem={}
def lemmatizza(text):
    if type(text) != int or type(text) != float:
        words= text.split(" ")
        stemmi= [it_stem.stem(x) for x in words if x not in stop_words]
    return stemmi

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

def emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def clean(series):
    series= series.str.lower()
    series=series.dropna()
    for i in "“#‼$%&!'-★”’•()*+,-™./:;<=>?@[\\]☑🤣😂^🇮🇹🤦👏🔴⚠_`{|}\n\t\"":
        series = series.str.replace(i, " ")
    series = series.str.replace("  ", "#")
    series = series.str.replace("#", " ")
    series = series.str.replace("  ", "#")
    series = series.str.replace("#", " ")
    #print(len(series.str.contains("  ")))
    return series

def convert_list_to_string(org_list, seperator=' '):
    return seperator.join(org_list)

try:
    path=sys.argv[1]
    if path[-1]!="\\" and path[-1]!="/":
        if "/" in path:
            path=path+"/"
        if "\\" in path:
            path=path+"\\"
except:
    path=''

#nlp = spacy.load("it_core_news_sm")
nlp = spacy.load("en_core_web_sm")
files=glob.glob(path+"*_aggregato.csv")
print (files)
diz = {}
#translator = google_translator()


for file in files:
    print(file)
    nomefile = file.split(".")[0]
    luogo = nomefile.split("_")[0].lower()
    print(luogo)
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_csv(file, sep=";")
    #df['english'] = df['Message'].apply(translator.translate, lang_src='it', lang_tgt='en')
    #print(df['english'])
    #print(df.columns)

    message_raw = df["title"]
    message = clean(message_raw)
    #print(message)
    #input()

    my_list = []
    for text in message:
        text = ''.join(c for c in text if not c.isnumeric())
        try:
            aux = text.decode().split()
        except:
            aux = text.split()
        for i in aux:
            if i not in my_list and i not in stop_words:
                my_list.append(i.lower())
    #print(my_list)

    aux = pd.DataFrame(my_list, columns=['word'])
    aux['word_stemmed'] = aux['word'].apply(lambda x: it_stem.stem(x))
    aux = aux.groupby('word_stemmed').transform(lambda x: ', '.join(x))
    aux['word_stemmed'] = aux['word'].apply(lambda x: it_stem.stem(x.split(',')[0]))
    aux.index = aux['word_stemmed']
    del aux['word_stemmed']
    my_dict = aux.to_dict('dict')['word']
    #print(my_dict)
    diz.update(my_dict)
    #print(diz)'''

    #translator.translate(msg for msg in message)

    lista_tokenizzata=[]
    for msg in message:
        #msg=emoji(msg)
        lista_tokenizzata.append(lemmatizza(msg))

    #wordcloud
    text = " ".join(msg for msg in message)
    WC_height = 500
    WC_width = 800
    WC_max_words = 100
    wordcloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=stop_words, background_color="white").generate(text)

    # Display the generated image:
    # the matplotlib way:
    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()
    wordcloud.to_file(luogo + "_wordcloud.png")


    bigram = gensim.models.Phrases(lista_tokenizzata, min_count=2, threshold=100)
    trigram = gensim.models.Phrases(bigram[lista_tokenizzata], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Form Bigrams
    data_words_bigrams = make_bigrams(lista_tokenizzata)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(lista_tokenizzata, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'])
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus [lista] di testo (texts) composti da lemmi (text)
    texts = data_lemmatized
    #converto la lista in text per fare una tf idf raw
    tokenized_list = [' '.join(inner_list) for inner_list in lista_tokenizzata]

    ###FREQUENZE RELATIVE###

    # tfid singole parole
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    X1_single = vectorizer.fit_transform(tokenized_list)
    features_single = (vectorizer.get_feature_names())
    # Applying TFIDF
    # You can still get n-grams here
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), norm='l2')
    X2_single = vectorizer.fit_transform(tokenized_list)
    #scores_single = (X2_single.toarray())
    #print(scores_single)
    # Getting top ranking features
    sums = X2_single.mean(axis=0)
    data1_single = []
    for col, term in enumerate(features_single):
        data1_single.append((term, sums[0, col]))
    ranking_single = pd.DataFrame(data1_single, columns=['term', 'rank'])
    words_single = (ranking_single.sort_values('rank', ascending=False))
    print("\n\nWords : \n", words_single.head(7))
    #prova somma per le singole parole
    #DF_TFIDF = pd.DataFrame(data=X2_single.toarray(), columns=features_single)
    #DF_TFIDF.to_csv("prova conto.csv")
    words_single.to_excel( luogo + "_relative_word_freq.xlsx")
    #words_single.T.sum(axis=1)
    tuples = [tuple(x) for x in words_single.values]
    WC_height = 500
    WC_width = 800
    Cloud = WordCloud(background_color="white", max_words=100, height=WC_height, width=WC_width).generate_from_frequencies(dict(tuples))
    Cloud.to_file(luogo + '_wordcloud_relative_words.jpg')

    #creating  tfid fpr trigrams
    vectorizer_tr = CountVectorizer(ngram_range=(3, 3))
    X1_trigr = vectorizer_tr.fit_transform(tokenized_list)
    features_trigr = (vectorizer_tr.get_feature_names())
    # Applying TFIDF
    vectorizer_trigr = TfidfVectorizer(ngram_range=(3, 3), norm='l2')
    X2_trigr = vectorizer_trigr.fit_transform(tokenized_list)
    #scores = (X2_trigr.toarray())
    #print(scores)
    # Getting top ranking features
    sums_trigr = X2_trigr.mean(axis=0)
    data1_trigr = []
    for col, term in enumerate(features_trigr):
        data1_trigr.append((term, sums_trigr[0, col]))
    ranking_trigr = pd.DataFrame(data1_trigr, columns=['term', 'rank'])
    words_trigr = (ranking_trigr.sort_values('rank', ascending=False))
    print("\n\nWords head : \n", words_trigr.head(7))
    words_trigr.to_excel(luogo + "_relative_trigrams.xlsx")
    tuples_trigr = [tuple(x) for x in words_trigr.values]
    WC_height = 500
    WC_width = 800
    Cloud = WordCloud(background_color="white", max_words=100, height=WC_height,
                      width=WC_width).generate_from_frequencies(dict(tuples_trigr))
    Cloud.to_file(luogo + '_wordcloud_relative_tirgrams.jpg')

    #creating  tfid for bigrams
    vectorizer_bi = CountVectorizer(ngram_range=(2, 2))
    X1 = vectorizer_bi.fit_transform(tokenized_list)
    features = (vectorizer_bi.get_feature_names())
    # Applying TFIDF
    # You can still get n-grams here
    vectorizer_bigr = TfidfVectorizer(ngram_range=(2, 2), norm='l2')
    X2 = vectorizer_bigr.fit_transform(tokenized_list)
    #scores_bigr = (X2.toarray())
    #print(scores_bigr)
    # Getting top ranking features
    sums = X2.mean(axis=0)
    data1 = []
    for col, term in enumerate(features):
        data1.append((term, sums[0, col]))
    ranking = pd.DataFrame(data1, columns=['term', 'rank'])
    words_big = (ranking.sort_values('rank', ascending=False))
    print("\n\nWords : \n", words_big.head(7))
    words_big.to_excel(luogo + "_relative_bigrams.xlsx")
    tuples_big = [tuple(x) for x in words_big.values]
    WC_height = 500
    WC_width = 800
    Cloud = WordCloud(background_color="white", max_words=100, height=WC_height,
                      width=WC_width).generate_from_frequencies(dict(tuples_big))
    Cloud.to_file(luogo + '_wordcloud_relative_bigrams.jpg')

    ###FREQENZE ASSOLUTE###
    #freqdist bigrammi
    testo= [y for x in texts for y in x]

    # PLOT E COUNT DELLE WORD CLOUD PER BIGRAMMI
    Bigrams = nltk.bigrams(testo)
    Frequenze_big = FreqDist(Bigrams)
    rslt_bi = pd.DataFrame(Frequenze_big.most_common(80), columns=['Bigrams', 'Frequency'])
    rslt_bi.to_excel(luogo + '_bigram.xlsx')
    #word cloud
    bigrams_list = list(nltk.bigrams(testo))
    #print(bigrams_list)
    dictionary2 = [' '.join(tup) for tup in bigrams_list]
    #print(dictionary2)

    # Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    #vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    #print(words_freq[:100])
    # Generating wordcloud and saving as jpg image
    words_dict = dict(words_freq)
    WC_height = 500
    WC_width = 800
    WC_max_words = 100
    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=stop_words, background_color="white")
    wordCloud.generate_from_frequencies(words_dict)
    #plt.title('Most frequently occurring bigrams connected by same colour and font size')
    #plt.imshow(wordCloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()
    wordCloud.to_file(luogo + '_wordcloud_bigram.jpg')

    # PLOT DELLE WORD CLOUD PER TRIGRAMMI
    Trigrams = nltk.trigrams(testo)
    Frequenze_trig = FreqDist(Trigrams)
    # print(Frequenze_trig.most_common(20))
    rslt_tr = pd.DataFrame(Frequenze_trig.most_common(80), columns=['Trigrams', 'Frequency'])
    rslt_tr.to_excel(luogo + '_trigram.xlsx')

    trigrams_list = list(nltk.trigrams(testo))
    # print(bigrams_list)
    dictionary3 = [' '.join(tup) for tup in trigrams_list]
    #print(dictionary3)

    # Using count vectoriser to view the frequency of bigrams
    vectorizer_3 = CountVectorizer(ngram_range=(3, 3))
    bag_of_words_3 = vectorizer_3.fit_transform(dictionary3)
    # vectorizer.vocabulary_
    sum_words_3 = bag_of_words_3.sum(axis=0)
    words_freq_3 = [(word, sum_words_3[0, idx]) for word, idx in vectorizer_3.vocabulary_.items()]
    words_freq_3 = sorted(words_freq_3, key=lambda x: x[1], reverse=True)
    #print(words_freq_3[:100])
    # Generating wordcloud and saving as jpg image
    trigr_dict = dict(words_freq_3)
    WC_height = 500
    WC_width = 800
    WC_max_words = 100
    wordcloud_trigr = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=stop_words, background_color="white")
    wordcloud_trigr.generate_from_frequencies(trigr_dict)
    plt.title('Most frequently occurring trigrams connected by same colour and font size')
    #plt.imshow(wordcloud_trigr, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()
    wordcloud_trigr.to_file(luogo + '_wordcloud_trigram.jpg')

    #FREQ ASSOLUTA SINGOLE PAROLE
    top_N = 150
    word_dist = nltk.FreqDist(testo)
    rslt = pd.DataFrame(word_dist.most_common(top_N), columns=['Word', 'Frequency'])
    rslt.to_excel(luogo + '_abs_words_frequency.xlsx')
    #print(rslt)


try:
    df_lems= pd.DataFrame.from_dict(diz, orient='index')
    print(df_lems)
    df_lems.to_csv("df_lemmi.csv", encoding= "utf-8")
except:
    pass

