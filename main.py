import nltk
#nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import findspark
findspark.init()
findspark.find()
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pandas as pd
import glob
import os
conf = pyspark.SparkConf().setAppName('te').setMaster('Local')
sc = pyspark.SparkContext()
spark = SparkSession(sc)
from pyspark.sql import SQLContext
lemmatizer = WordNetLemmatizer()
tokens = []
words = []
porter = PorterStemmer()
documents = sc.wholeTextFiles("./dataset/*").map(lambda x: x[1]).collect()
for x in range(len(documents)):
    documents[x] = documents[x].replace('\n\n\n',' ').replace('\n\n',' ').replace('\n',' ')
for x in range(len(documents)):
    tk = word_tokenize(documents[x])
    tk = [w.lower() for w in tk]
    tokens.append(tk)
    wds = [word for word in tokens[x] if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    wds = [stopword for stopword in wds if not stopword in stop_words]
    #wds = [porter.stem(word) for word in wds]
    wds = [lemmatizer.lemmatize(word) for word in wds]
    wds = list(dict.fromkeys(wds))
    words.append(wds)
DF = {}
for x in range(len(words)):
    z = words[x]
    for y in z:
        try:
            DF[y].add(x)
        except:
            DF[y] = {x}
identifiers = {}
idx = 0
for word in unique:
    if word not in identifiers:
        identifiers[word] = idx
        idx += 1
{x: DF.get(x, x) for x, y in identifiers.items()}
