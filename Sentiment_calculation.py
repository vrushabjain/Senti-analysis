# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:16:30 2017

@author: Vrushab PC
"""
from textblob import TextBlob
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

con = sqlite3.connect('database.sqlite') #getting data from database

allmail = pd.read_sql_query("""
        SELECT * FROM Emails
        where strftime('%Y', MetadataDateSent) = '2010'
        """, con) # sql query to extract data for year '2009'. Change it to '2010' and '2012' to get results for these years

countries = ['Libya']  #change name of countries to get sentiment scores for each countries

filtered = []   # appending all emails containg that specific countries
for mail in allmail['ExtractedBodyText']: # emails are present in this column
    if any(word in mail for word in countries):
        
        filtered.append(mail)

mail = allmail[allmail['ExtractedBodyText'].isin(filtered)]
#mail = mail[mail['MetadataFrom'] == 'H']

print("Number of Emails?") 
print(mail['ExtractedBodyText'].size) #Finding number of emails for that particular country
###############
##sentiment code built using textblob sentiment analyzer documnetation
###############
empolarity = []
emsubject = []
for row in mail['ExtractedBodyText']:
    toput = TextBlob(row)
    empolarity.append(toput.sentiment.polarity)
    emsubject.append(toput.sentiment.subjectivity)

mail['Polarity'] = empolarity
mail['Subjectivity'] = emsubject

sentiment_scores = mail.sort_values(by = 'Polarity', ascending = False)[['MetadataDateSent', 'Polarity', 'Subjectivity', 'ExtractedBodyText', 'RawText']]
# code below gives the most positive email and most negative email for that particular country
print("Topmost positive Email is:")
print("'" + sentiment_scores['ExtractedBodyText'].iloc[0] + "'")
print("It has a polarity of {0}".format(sentiment_scores['Polarity'].iloc[0]) + ",")
print("with a subjectivity of {0}".format(sentiment_scores['Subjectivity'].iloc[0]) + ",")
print("but the statement is more neutral than anything else.")

sentiment_scores = mail.sort_values(by = 'Polarity', ascending = True)[['MetadataDateSent', 'Polarity', 'Subjectivity', 'ExtractedBodyText', 'RawText']]

print("Topmost negative Email is as:")
print("'" + sentiment_scores['ExtractedBodyText'].iloc[0] + "'")
print("It has a polarity of {0}".format(sentiment_scores['Polarity'].iloc[0]) + ",")
print("with a subjectivity of {0}".format(sentiment_scores['Subjectivity'].iloc[0]) + ".")

average = sum(mail['Polarity']) / float(len(mail['Polarity'])) #calculating the overall sentiment for that country
print("The overall sentiment of the country", average)   

