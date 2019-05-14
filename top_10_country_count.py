# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:12:39 2017

@author: Vrushab PC
"""
import sqlite3
from collections import Counter, defaultdict

# Data Cleanup & Exploratory Data Analysis
# import email  - didn't work on raw email txt
import re
import numpy as np
import pandas as pd
import ftfy
import pycountry




# Visualization
import matplotlib.pyplot as plt
import seaborn


con = sqlite3.connect('database.sqlite')

emails = pd.read_sql_query("SELECT * FROM Emails", con)

emails.info()

emails.MetadataDateSent = pd.to_datetime(emails.MetadataDateSent)
emails['Year'] = emails['MetadataDateSent'].dt.year
emails['Month'] = emails['MetadataDateSent'].dt.month

emails = emails.drop(['MetadataPdfLink','DocNumber', 'ExtractedDocNumber', 'MetadataCaseNumber'], axis=1)

emails = emails[emails.RawText.str.len()>10]

countries_lookup = {}

# Grab countries objects
countries = list(pycountry.countries)

# Fill dict with [2 letter codes] : country names
for country in countries:
    countries_lookup[country.alpha_2]=country.name

country_names = list(countries_lookup.values())
country_names = [country.lower() for country in country_names]
country_names = set(country_names)
country_names.add('syria')

def country_counts(emails, country_names):
    country_mentions = defaultdict(int)
    for email in emails:
        email_tokens = email.split()
        if 'libyan' in email_tokens:
            country_mentions['libya'] += 1
        elif 'syrian' in email_tokens:
            country_mentions['syria'] += 1
        else:
            intersect = country_names.intersection(email_tokens)
            for match in intersect:
                    country_mentions[match] += 1
    return country_mentions

country_test = emails.RawText
c_mentioned = Counter(country_counts(country_test, country_names))

c_counts = [value[1] for value in c_mentioned.most_common(10)]
c_countries = [value[0] for value in c_mentioned.most_common(10)]

plt.figure(figsize=(12,6))

plt.bar(range(len(c_counts)), c_counts, color='orange')
plt.xticks(range(len(c_counts)), c_countries, rotation=45, ha='center')


plt.xlabel('Top 10 Countries Mentioned')
plt.ylabel('Mentions')
plt.show()



