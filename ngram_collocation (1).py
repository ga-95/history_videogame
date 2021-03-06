import collocations
import nltk
from nltk.probability import FreqDist
from nltk import stem
it_stem=nltk.stem.SnowballStemmer('english')
nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint
import numbers
import networkx as nx
from creagrafico import crea_grafico
import collocations


import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend([ 'u', 'https', 'www', 'youtube',  'com', 'removed', 'http',  'wikipedia', 'kotakuinact','9z3vqo', 'en', '[deleted]',
                    'org', 'wiki', 'rep', 'like', 'wikia', 'youtub', 'yldaukrnl2q', 'r', 'KotakuInAction', 'kotakuinaction'])
#'would', 'want', 'could', 'go', 'get',


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
    for i in "“#‼$%&!'-★”’•()*+,-./:;<=>?@[\\]☑🤣😂^🇮🇹🤦👏🔴⚠_`{|}\n\t\"":
        series = series.str.replace(i, " ")
    series = series.str.replace("  ", "#")
    series = series.str.replace("#", " ")
    series = series.str.replace("  ", "#")
    series = series.str.replace("#", " ")
    #print(len(series.str.contains("  ")))
    return series


diz_lem={}
def lemmatizza(text):
    if type(text) != int or type(text) != float:
        words= text.split(" ")
        stemmi= [it_stem.stem(x) for x in words if x not in stop_words]
    return stemmi


df=pd.read_csv('wf_aggregato.csv', encoding="utf-8" )
df["mes_token"] = " "
df["clean_message"] = " "
db_tot=pd.DataFrame()
#print(df.columns)
print(df.shape)

message_raw = df["title"]
df["clean_message"] = clean(message_raw)
message = clean(message_raw)
#print(df["clean_message"])

lista_tokenizzata=[]
for msg in message:
    lista_tokenizzata.append(lemmatizza(msg))

#print(lista_tokenizzata)
print(len(lista_tokenizzata))
print(len(message))

df['mes_token']=lista_tokenizzata

#print(df['mes_token'])

token= "jew"
my_res = collocations.collocations([token], df, 5, text_column="mes_token")

df1= my_res[0]
df1.to_csv("wf_" + token + "_abs_col_gen.csv")
print(df1)
df2= my_res[1]
df2.to_csv("wf_" + token + "_rel_col_gen.csv")
print(df2)


words = list(df2.index)
print(words)
collo=collocations.collocations(words, df, 5, text_column="mes_token")
df_collocato= collo[1]
df_collocato.to_csv("wf_" + token + "_collocazioni_multiple.csv")

df_collocato= pd.read_csv("wf_jew_collocazioni_multiple.csv", index_col= "ITEM")
print(df_collocato)
try:
    df_collocato.drop("ITEM.1", axis=1, inplace=True)
except:
    pass
print(df_collocato)
#input()
nx.write_gexf(crea_grafico(df_collocato), "GRAFICO_jew.gexf")