
# coding: utf-8

# In[372]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn
import datetime as dt
from wordcloud import WordCloud

import heapq
import collections

import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer

import stemming
from stemming import porter2

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

import scipy
import scipy.interpolate as sc_int
import scipy.sparse as sc_sp

from pprint import pprint

from time import time

from IPython.display import display


# In[2]:


# %matplotlib inline


# # Evolve Interview Project
# 
# ## Zoë Farmer
# 
# <img src="https://images.fineartamerica.com/images-medium-large-5/boston-back-bay-skyline-at-night-color-panorama-jon-holiday.jpg" style="width: 100%;" />
# 
# *Press the space-bar to proceed to the next slide. [See here for a brief tutorial](http://lab.hakim.se/reveal-js/)*

# ## Who am I?
# 
# * My name is Zoë Farmer
# * Recent CU graduate with a BS in Applied Math and a CS Minor
# * Co-coordinator of the Boulder Python Meetup
# * Big fan of open source software
# * http://www.dataleek.io
# * [@thedataleek](http://www.twitter.com/thedataleek)
# * [git(hub|lab).com/thedataleek](http://github.com/thedataleek)

# ## General Tooling Overview
# 
# * Everything is in Python3.6
# * I use `jupyter`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `nltk`, and `scipy`.
# * Some code has been skipped for brevity. See [this link](http://www.gitlab.com/thedataleek/evolve_interview) for full code.
# * Development performed with Jupyter Notebook, this notebook is available at the above link.
# * Presentation powered by Reveal.js
# 
# <div>
#     <div style="display: inline-block; width: 16%;"><img src="./evolve_interview/jupyter.svg"/></div>
#     <div style="display: inline-block; width: 20%;"><img src="./evolve_interview/pandas.png"/></div>
#     <div style="display: inline-block; width: 12%;"><img src="./evolve_interview/numpy.jpg"/></div>
#     <div style="display: inline-block; width: 18%;"><img src="./evolve_interview/matplotlib.svg"/></div>
#     <div style="display: inline-block; width: 12%;"><img src="./evolve_interview/sklearn.png"/></div>
#     <div style="display: inline-block; width: 8%;"><img src="./evolve_interview/scipy.png"/></div>
# </div>

# # The Data

# ## What is it?
# 
# A year of data about Boston scraped from AirBnB that contains 2 datasets
# 
# * listing details
# * calendar information

# ## (1) Listings - details about locations
# 
# Our first dataset is a large number of listings and associated descriptions.

# In[195]:


listing_data = pd.read_csv('./ListingsAirbnbScrapeExam.csv')


# In[196]:


len(listing_data)


# In[197]:


', '.join(listing_data.columns)


# ## (2) Calendar Data - location occupancy by date
# 
# Our second dataset is a set of listings by date, occupancy, and price.
# 
# * We want to parse these fields
#     * datestrings to be formatted as python `datetime` objects
#     * price field to be floats

# In[198]:


price_re = '^ *\$([0-9]+\.[0-9]{2}) *$'
def price_converter(s):
    match = re.match(price_re, s)
    if match:
        return float(match[1])
    else:
        return np.nan


# In[199]:


calendar_data = pd.read_csv(
    './CalendarAirbnbScrapeExam.csv',
    converters={
        'available': lambda x: True if x == 'f' else False,
        'price': price_converter
    }
)
calendar_data['filled'] = ~calendar_data['available']
calendar_data['date'] = pd.to_datetime(calendar_data['date'],
                                       infer_datetime_format=True)


# In[200]:


calendar_data.head(1)


# # Dataset Merge

# We want to combine datasets
# 
# * Let's calculate the number of nights occupied per listing and add to the listing data.
# * Average/standard deviation price per night
# 
# But let's first make sure the datasets overlap.

# In[201]:


listing_keys = set(listing_data.id)
calendar_keys = set(calendar_data.listing_id)
difference = listing_keys.difference(calendar_keys)
print(f'# Listing Keys: {len(listing_keys)}')
print(f'# Calendar Keys: {len(calendar_keys)}')
print(f'# Difference: {len(difference)}')


# They don't, in fact we're missing information on about 700 listings.
# 
# For our `num_filled` column let's establish the assumption that a `NaN` value stands for "unknown".

# ## Groupby
# 
# We can simply `sum()` our `available` and `filled` boolean fields. This will give us a total number of nights occupied (or available).
# 
# Note, in the final aggregated sum these two fields sum to 365.

# In[202]:


fill_dates = calendar_data    .groupby('listing_id')[['available', 'filled', 'price']]    .agg({
        'available': 'sum',
        'filled': 'sum',
        'price': ['mean', 'std']
    })
fill_dates['listing_id'] = fill_dates.index


# In[203]:


fill_dates.head()


# ## Left Join
# 
# Now we merge with our original dataset using a left join.

# In[204]:


combined_data = listing_data.merge(
    fill_dates,
    how='left',
    left_on='id',
    right_on='listing_id'
)
combined_data.rename(
    columns={
        ('available', 'sum'): 'available',
        ('filled', 'sum'): 'filled',
        ('price', 'mean'): 'avg_price',
        ('price', 'std'): 'std_price'
    },
    inplace=True
)


# In[205]:


# make sure that merge worked the way we want it to
for key in listing_data.id:
    combined_val = combined_data[combined_data.id == key][['available']].values
    fill_val = fill_dates[fill_dates.listing_id == key][['available']].values
    try:
        assert combined_val == fill_val
    except AssertionError:
        if np.isnan(combined_val[0, 0]) and len(fill_val) == 0:
            continue
        else:
            print(key, combined_val, fill_val)
            raise
for key in difference:
    assert np.isnan(combined_data[combined_data.id == key][['available']].values[0, 0])
    assert len(fill_dates[fill_dates.listing_id == key]) == 0


# In[206]:


combined_data[['id', 'name', 'available', 'avg_price', 'std_price']].head(10)


# ## Neighborhood Statistics
# 
# Now that we've added those columns to the listing data, we can start to get neighborhood-specific statistics.

# In[239]:


valid_combined = combined_data[~combined_data['available'].isnull()]
neighborhood_stats = valid_combined    .groupby('neighbourhood_cleansed')    .agg({
        'avg_price': 'mean',
        'std_price': 'mean'
    }
)
neighborhood_stats.sort_values('avg_price', inplace=True)
neighborhood_stats.plot(kind='bar', figsize=(12, 6))
plt.title('Neighborhood Price and Standard Deviation')
plt.xlabel('Neighborhood')
plt.ylabel('Cost (Dollars)')
plt.tight_layout()
plt.savefig('./evolve_interview/neighborhood_stats.png')


# <img src="./evolve_interview/neighborhood_stats.png" style="width: 100%; height: 50%;" />

# # Seasonal Trends

# We have a year of data, let's examine how seasons effect occupancy.
# 
# We can take a naive approach and simply `groupby` each date and plot the number of dates filled. 

# In[241]:


calendar_data.groupby('date')[['filled']].sum().plot(figsize=(12, 6))
plt.title('Total Occupancy per Day')
plt.xlabel('Date')
plt.ylabel('Total Occupancy')
plt.tight_layout()
plt.savefig('./evolve_interview/naive_occupancy.png')


# <img src="./evolve_interview/naive_occupancy.png" />

# ## Let's do better
# 
# This chart has some irregularities and is a little unclear about the type of trends we're looking for.
# 
# Let's look at only the listings that are filled each day of the year, and look at their prices as the year goes by.
# 
# We'll refer to these as "indicator listings".

# In[209]:


days_filled = calendar_data.groupby('listing_id')[['filled']].sum()
top_listings = days_filled[days_filled.filled == 365].index


# In[210]:


print(f'Original Datasize: {len(calendar_data.listing_id.unique())}.')
print(f'Pruned Datasize: {len(top_listings)}')


# This shrinks our dataset by a lot, but that's ok.
# 
# We're looking for indicator listings, not the entire dataset.

# In[211]:


pruned_calendar_data = calendar_data[
    calendar_data['listing_id'].isin(top_listings)
]


# ## Plotting our Busy Listings

# In[242]:


plt.figure(figsize=(12, 6))
for lid in top_listings:
    cdata = pruned_calendar_data[pruned_calendar_data.listing_id == lid]
    cdata = cdata.sort_values('date')
    plt.plot(cdata.date, cdata.price)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Occupied Listing Price per Day')
plt.tight_layout()
plt.savefig('./evolve_interview/all_filled.png')


# <img src="./evolve_interview/all_filled.png" />

# ## Reducing Noise
# 
# This chart has too much noise and the trends are even less clear.
# 
# * Remove all listings with low standard deviation
#     * $10 < \sigma < 200$
# * Also cut out all listings that only have a few unique values
#     * $\left\lvert \left\{ X \right\}\right\rvert > 10$
# * Periodicity is the enemy of seasonal trends

# In[213]:


listing_price_deviations = pruned_calendar_data.groupby('listing_id')[['price']].std()
listing_price_deviations.rename(columns={'price': 'stddev'}, inplace=True)
listing_price_deviations = listing_price_deviations[
    np.logical_and(
        listing_price_deviations.stddev > 10,
        listing_price_deviations.stddev < 200
    )
]
listing_periodicity = pruned_calendar_data.groupby('listing_id')[['price']].nunique()
listing_periodicity.rename(columns={'price': 'num_unique'}, inplace=True)
listing_periodicity = listing_periodicity[
    listing_periodicity.num_unique > 10
]
sensitive_listings = listing_price_deviations.join(listing_periodicity, how='inner')


# In[214]:


sensitive_listings


# In[215]:


sensitive_listings = sensitive_listings.index
sensitive_calendar_data = calendar_data[calendar_data['listing_id'].isin(sensitive_listings)]


# In[243]:


plt.figure(figsize=(12, 6))
for lid in sensitive_listings:
    cdata = sensitive_calendar_data[sensitive_calendar_data.listing_id == lid]
    cdata = cdata.sort_values('date')
    label = combined_data[combined_data.id == lid].name.values[0]
    plt.plot(cdata.date, cdata.price, label=label)
plt.legend(loc=0)
plt.title('Indicator Listing Prices by Day')
plt.xlabel('Date')
plt.ylabel('Price')
plt.tight_layout()
plt.savefig("./evolve_interview/indicators.png")


# ## Plotting our Indicator Listings
# 
# <img src="./evolve_interview/indicators.png" />

# In[245]:


fig, axarr = plt.subplots(2, 1, figsize=(16, 12))
for lid in sensitive_listings:
    cdata = sensitive_calendar_data[sensitive_calendar_data.listing_id == lid]
    cdata = cdata.sort_values('date')
    label = combined_data[combined_data.id == lid].name.values[0]
    axarr[0].plot(cdata.date.values, cdata.price.values, label=label)
axarr[0].legend(loc=0)
axarr[0].set_title('Indicator Listings')
axarr[0].set_xlabel('Date')
axarr[0].set_ylabel('Price')

fill_dates = calendar_data.groupby('date')[['filled']].sum()
axarr[1].plot(fill_dates.index.values, fill_dates.filled.values)
axarr[1].set_title('Total Occupancy')
axarr[1].set_xlabel('Date')
axarr[1].set_ylabel('Occupancy')

plt.tight_layout()
plt.savefig("./evolve_interview/indicators_occupancy.png")


# ## Combining Naive Occupancy and Indicator Listings
# 
# <img src="./evolve_interview/indicators_occupancy.png" />

# ## What does this tell us?
# 
# * Winter was the busy season for 2016-2017
#     * Most likely because of family/holidays
#     * Also the cheapest
# * Summers are expensive
# * Memorial Day Weekend is expensive (the spike in the middle)
#     * [See this event guide for details](http://www.boston-discovery-guide.com/boston-event-calendar-may.html)
# * The start of MIT school year is expensive (spike at the right side)
#     * [See the academic calendar for more info](http://web.mit.edu/registrar/calendar/)
# * Visit Boston between New Years and March for the cheapest rates.
# * Weekends are more expensive than weekdays, but this doesn't influence occupancy.
# * Our naive approach looks weird in Fall 2016 due to AirBnB's increased activity in the area
#     * [See here for 2016 article](http://www.bostonherald.com/news/local_coverage/2016/10/airbnb_leaving_no_room_for_rentals)
# * [According](https://www.bostonglobe.com/business/2017/08/06/cambridge-set-vote-ordinance-regulate-airbnb/61Wc1phpViVbYv3x7ORMaI/story.html) to [a ton](https://www.bostonglobe.com/business/2017/10/05/housing-advocates-say-airbnb-rentals-are-replacing-chinatown-apartments/Z9KwIgppY89rHbbflvKC6H/story.html) of [news sources](http://www.wbur.org/bostonomix/2017/08/08/cambridge-short-term-rental-rules), this was an year of protest for AirBnB. This is probably skewing the data
#     

# These are good preliminary results, but for more accurate results we'd want several years to reduce influence of increased activity, year specific events, legal actions, etc.

# # Neighborhood Specific Seasonal Trends

# Let's dig into any seasonal trends we can find on a neighboorhood basis.

# In[218]:


full_combined_data = listing_data.merge(
    calendar_data,
    how='inner',
    left_on='id',
    right_on='listing_id'
)


# Let's plot each neighborhood by their average price and fill-rate per day.

# In[219]:


full_combined_data = full_combined_data[~full_combined_data.available.isnull()]


# In[220]:


neighborhood_data = full_combined_data    .groupby(['neighbourhood_cleansed', 'date'])    .agg({'filled': 'sum', 'price_y': 'mean'})
neighborhood_data = neighborhood_data.unstack(level=0)


# In[246]:


neighborhood_data[['filled']].plot(
    figsize=(12, 8),
    legend=False
)

plt.ylabel('Occupancy')

plt.title('Neighborhood Occupancy')
plt.savefig('./evolve_interview/neighborhood_filled.png')

neighborhood_data[['price_y']].plot(
    figsize=(12, 8),
    legend=False
)

plt.ylabel('Price')

plt.title('Neighborhood Average Price')
plt.savefig('./evolve_interview/neighborhood_price.png')


# <img src="./evolve_interview/neighborhood_filled.png" />

# <img src="./evolve_interview/neighborhood_price.png" />

# ## What does this tell us?
# 
# * As with before, Memorial Day Weekend stands out as a spike in pricing and a drop in occupancy
# * Weekends are more expensive
# * December and March 1st have a huge drop in occupancy and pricing
# * Not every seasonal trend affects every neighborhood! Some are immune (or do the opposite) of the average trend.

# As with before, we'd ideally want more data to make more accurate observations.

# # Examining Neighborhoods

# Let's also see if we can pull out neighbor features.
# 
# Some listings don't have neighborhood descriptions, so let's skip those.

# In[222]:


valid_desc_data = combined_data[combined_data.neighborhood_overview.notnull()].copy()
neighborhood_labels = valid_desc_data.neighbourhood_cleansed.unique()


# In[223]:


neighborhood_labels


# ## How many listings per neighborhood?

# In[224]:


valid_desc_data.groupby('neighbourhood_cleansed').agg('size').sort_values()


# In[225]:


top5_neighborhoods = list(valid_desc_data.groupby('neighbourhood_cleansed').agg('size').sort_values().tail(5).index)
top5_listings = valid_desc_data[valid_desc_data.neighbourhood_cleansed.isin(top5_neighborhoods)]
for key, group in top5_listings.groupby('neighbourhood_cleansed'):
    plt.figure()
    text = '\n'.join(group.neighborhood_overview.values)

    wordcloud = WordCloud(width=1600, height=1200).generate(text)
    plt.figure(figsize=(16, 12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(key)
    plt.axis("off")
    plt.savefig(f'./evolve_interview/{"_".join(key.lower().split(" "))}_words.png')


# ## Where are these neighborhoods?
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Boston_ONS_Neighborhoods.svg/1200px-Boston_ONS_Neighborhoods.svg.png" style="width: 50%"/>

# ## Top 5 Neighborhoods
# 
# Let's only take the top 5 neighborhoods with the most listings.

# In[226]:


top5_neighborhoods


# Now let's make a word cloud for each neighborhood based on the most common words in their descriptions.

# ## Allston
# 
# <img src="./evolve_interview/allston_words.png" />

# ## Fenway
# 
# <img src="./evolve_interview/fenway_words.png" />

# ## Back Bay
# 
# <img src="./evolve_interview/back_bay_words.png" />

# ## South End
# 
# <img src="./evolve_interview/south_end_words.png" />

# ## Jamaica Plain
# 
# <img src="./evolve_interview/jamaica_plain_words.png" />

# # Feature Extraction

# Wordclouds are pretty, but also fairly crude. Let's take a deeper dive into these top 5 neighborhoods.

# In[249]:


top5_fulltext = top5_listings[['neighbourhood_cleansed',
                               'neighborhood_overview']]
top5_fulltext.head(3)


# ## Term Frequency - Inverse Document Frequency
# 
# From [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf),
# 
# > tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpu
# 
# In essence, the product of "how common a word is in the corpus" and "inverse of how frequently the term appears in the document set".
# 
# Using this concept we can construct a document matrix, where each row represents a document in the corpus, and each column represents a word that appeared.
# 
# The big difference between this approach and our wordcloud approach from earlier which just relies on raw frequency is that this takes into account the overall frequency of the word in the entire document set.

# ## Scikit-Learn
# 
# `sklearn` provides several vectorizers, including a tf-idf vectorizer.
# 
# We give it a tokenizing regular expression in order to prune less relevant tokens (a token is just a unit of semantic meaning, in this case we only want words longer than 3 characters).

# In[373]:


valid_word = '[A-Za-z_]'
vect = TfidfVectorizer(
    token_pattern=f'(?u)\\b{valid_word}{{5,}}\\b'
)
stemmer = porter2
tokenizer = TreebankWordTokenizer()


# In[374]:


top5_cleantext = np.empty(len(top5_fulltext), dtype=object)
for i, text in enumerate(top5_fulltext.values[:, 1]):
    splittext = [x for x in text.split(' ')
                 if (len(x) > 3 and
                     not x[0].isupper())]
    top5_cleantext[i] = (' '.join(stemmer.stem(word) for word in tokenizer.tokenize(' '.join(splittext))))


# We're going to feed this all of the listing descriptions from our top 5 neighborhoods and aggregate later.

# In[375]:


fit = vect.fit(top5_cleantext)
X = vect.fit_transform(top5_cleantext)


# The shape of this document matrix, $946 \times 1599$, indicates there are $946$ documents, and $1599$ tokens.
# 
# This matrix is incredibly sparse however (only about 0.5% full), since not every token appears in every document.

# In[376]:


X.shape


# In[377]:


X.astype(bool).sum() / (946 * 3019)


# ## Using tf-idf
# 
# Now that we have our document matrix, let's use it to figure out the most important words per document.

# In[386]:


neighborhood_docs = {i: name for i, name in enumerate(top5_fulltext.values[:, 0])}
vocab = {v: k for k, v in fit.vocabulary_.items()}
words = {x: [] for x in set(top5_fulltext.values[:, 0])}
for (document, index), val in sc_sp.dok_matrix(X).items():
    n = neighborhood_docs[document]
    word = vocab[index]
    heapq.heappush(words[n], (val, word))
summary = ''
for neighborhood, wordlist in words.items():
    wordlist = wordlist[::-1]
    final_list = []
    for val, word in wordlist:
        if len(final_list) > 15:
            break
        if word not in final_list:
            final_list.append(word)
    summary += (f'{neighborhood}:\n\t{", ".join(final_list)}\n\n')


# In[387]:


print(summary)


# ## What does this tell us?
# 
# * Stemming (converting words to their "base" form) is tricky, and innacurate
# * Tf-Idf emphasizes words that appear in fewer documents
#     * This gives us a better summary instead of just seeing "Boston" for every neighborhood
# * The advantage this provides over just word frequencies is that we see the important things that aren't mentioned frequently.
# * South End:
# 	* Located in a good spot, younger crowd, good restaurants, "deeper beauty".
# * Fenway
# 	* Younger crowd, has museums, multicultural, modern.
# * Back Bay:
# 	* Hospital access, conventions here, high value, historical districts
# * Jamaica Plain:
# 	* Lots of zagat-reviewed restaurants, good food here, younger crowd.
# * Allston:
# 	* Younger crowd, access to outdoors activities (biking, etc.), active nightlife.

# # Conclusions

# ## Seasonal Trends
# 
# <img src="./evolve_interview/indicators_occupancy.png" />

# * Winter was the busy season for 2016-2017
#     * Most likely because of family/holidays
#     * Also the cheapest
# * Summers are expensive
# * Memorial Day Weekend is expensive (the spike in the middle)
#     * [See this event guide for details](http://www.boston-discovery-guide.com/boston-event-calendar-may.html)
# * The start of MIT school year is expensive (spike at the right side)
#     * [See the academic calendar for more info](http://web.mit.edu/registrar/calendar/)
# * Visit Boston between New Years and March for the cheapest rates.
# * Weekends are more expensive than weekdays, but this doesn't influence occupancy.
# * Our naive approach looks weird in Fall 2016 due to AirBnB's increased activity in the area
#     * [See here for 2016 article](http://www.bostonherald.com/news/local_coverage/2016/10/airbnb_leaving_no_room_for_rentals)
# * [According](https://www.bostonglobe.com/business/2017/08/06/cambridge-set-vote-ordinance-regulate-airbnb/61Wc1phpViVbYv3x7ORMaI/story.html) to [a ton](https://www.bostonglobe.com/business/2017/10/05/housing-advocates-say-airbnb-rentals-are-replacing-chinatown-apartments/Z9KwIgppY89rHbbflvKC6H/story.html) of [news sources](http://www.wbur.org/bostonomix/2017/08/08/cambridge-short-term-rental-rules), this was an year of protest for AirBnB. This is probably skewing the data
#   

# <img src="./evolve_interview/neighborhood_filled.png" style="display: inline-block; width: 50%; height: 50%" />
# <img src="./evolve_interview/neighborhood_price.png" style="display: inline-block; width: 49%; height: 49%" />

# * As with before, Memorial Day Weekend stands out as a spike in pricing and a drop in occupancy
# * Weekends are more expensive
# * December and March 1st have a huge drop in occupancy and pricing
# * Not every seasonal trend affects every neighborhood! Some are immune (or do the opposite) of the average trend.

# ## Neighborhoods

# The Leather District, West End, and Chinatown are the most expensive places to live.
# 
# <img src="./evolve_interview/neighborhood_stats.png" style="width: 100%; height: 50%;" />

# <img src="./evolve_interview/allston_words.png" style="display: inline-block; width: 30%" />
# <img src="./evolve_interview/fenway_words.png" style="display: inline-block; width: 30%" />
# <img src="./evolve_interview/back_bay_words.png" style="display: inline-block; width: 30%" />
# <img src="./evolve_interview/south_end_words.png" style="display: inline-block; width: 30%" />
# <img src="./evolve_interview/jamaica_plain_words.png" style="display: inline-block; width: 30%" />

# * South End:
# 	* Located in a good spot, younger crowd, good restaurants, "deeper beauty".
# * Fenway
# 	* Younger crowd, has museums, multicultural, modern.
# * Back Bay:
# 	* Hospital access, conventions here, high value, historical districts
# * Jamaica Plain:
# 	* Lots of zagat-reviewed restaurants, good food here, younger crowd.
# * Allston:
# 	* Younger crowd, access to outdoors activities (biking, etc.), active nightlife.

# # Questions?
# 
# * [zoe@dataleek.io](mailto:zoe@dataleek.io)
# * [https://gitlab.com/thedataleek/evolve_interview](https://gitlab.com/thedataleek/evolve_interview)
