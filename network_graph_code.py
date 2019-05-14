# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:40:58 2017

@author: Vrushab PC
"""

import sqlite3
import pandas as pd
import numpy as np
con = sqlite3.connect('database.sqlite')

emails = pd.read_sql_query("""
SELECT p.Name Sender,
       e.SenderPersonId Id_sender, e.MetadataTo, e.ExtractedBodyText text,
       a.PersonId
FROM Emails e
INNER JOIN Persons p ON e.SenderPersonId=p.Id 
LEFT OUTER JOIN Aliases a ON lower(e.MetadataTo)=a.Alias
""", con) ####get the data from the sql query

persons = pd.read_sql_query("""
SELECT Id, Name
FROM Persons 
""", con)
personsDict = {}
for i in persons.values:
    personsDict[i[0]] = i[1]
####
## code used from https://www.kaggle.com/gpayen/interaction-between-contacts
####
def computeSender(item):
    # Sender is Hillary Clinton
    if item.Id_sender == 80 and item.MetadataTo != '' and np.isnan(item.PersonId):
        tab = item.MetadataTo.split(',')
        name = tab[1].strip() + ' ' + tab[0].strip() 
        tmp = pd.read_sql_query("SELECT Id, Name FROM Persons WHERE Name='"+ name +"'", con)
        # A person was found
        if not tmp.empty:
            item.PersonId = tmp['Id'][0]
    # Create the new Contact column
    if item.Id_sender == 80:
        item['Id_Contact'] = item.PersonId
    else:
        item['Id_Contact'] = item.Id_sender
    return item
print("Number of emails before cleaning : ",emails.shape[0])

data = emails.apply(computeSender, axis=1);

# Remove the not found persons
data = data[(~np.isnan(data.PersonId)) | (data.Id_sender != 80)]
data = data[data.Id_Contact != 80]
data['Id_Contact'] = data['Id_Contact'].apply(lambda i : personsDict[int(i)])



corpusTmp = {}
corpus = {}

for i, email in enumerate(data.values):
    corpusTmp[email[5]] = corpusTmp.get(email[5], "") + email[3]
    
occ = []
for key, val in corpusTmp.items():
    if int(len(val)) > 10:
        corpus[key] = val
contacts = list(corpus.keys())

import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for contactId, text in corpus.items():
    lowers = text.lower()
    no_punctuation = lowers.translate(string.punctuation)
    corpus[contactId] = no_punctuation
        
# because documents are of different size, it is important to normalize !
tf_idf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', norm='l2')
tf_idf_1 = tf_idf.fit_transform(corpus.values())

from sklearn.metrics.pairwise import cosine_similarity
import operator
similarity = cosine_similarity(tf_idf_1[0:len(contacts)], tf_idf_1)

links = {}
threshold = 0.5
limit = 5
x = []
contactsGraph = []

for i in range(len(contacts)):
    tmp = {}
    for j in range(i):
        if similarity[int(i),int(j)] > threshold:
            contactsGraph.append(int(i))
            contactsGraph.append(int(j))
            tmp["%s/%s" % (contacts[int(i)],contacts[int(j)])] = similarity[int(i),int(j)]
    tmp = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(limit):
        if i < len(tmp):
            links[tmp[i][0]] = tmp[i][1]
          

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (50, 50)

graph1=nx.Graph()

for key, val in links.items():
    n1, n2 = key.split('/')
    graph1.add_edge(n1, n2, weight=round(val, 3))

link = dict()
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    link[i] = [(u,v) for (u,v,d) in graph1.edges(data=True) if d['weight'] < i and d['weight'] >= round(i-0.1, 2)]

pos=nx.spring_layout(graph1)

nx.draw_networkx_nodes(graph1,pos,node_size=0,node_color='yellow')

for key, val in link.items():
    nx.draw_networkx_edges(graph1,pos,edgelist=val, width=1, style='dashed') 

nx.draw_networkx_labels(graph1,pos,font_size=18,font_family='sans-serif')

plt.axis('off')
plt.show()
