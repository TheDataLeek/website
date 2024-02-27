---
layout: post
title: Latent Semantic Analysis with Python
nav-menu: false
show_tile: false
---

In this project we will perform [latent semantic
analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) of large
document sets.

We first create a [document term
matrix](https://en.wikipedia.org/wiki/Document-term_matrix), and then perform
[SVD decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition).

This document term matrix uses
[tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weighting.


# Loading Data

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
import scipy.sparse.linalg as ssl
import scipy.sparse as scs
from tqdm import tqdm
```


```python
%load_ext cython
```

First we'll load in the data. Using the Jeopardy data as it's small.


```python
data = pd.read_csv('./data/JEOPARDY_CSV.csv')
data = data[:1000]
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Show Number</th>
      <th>Air Date</th>
      <th>Round</th>
      <th>Category</th>
      <th>Value</th>
      <th>Question</th>
      <th>Answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>HISTORY</td>
      <td>$200</td>
      <td>For the last 8 years of his life, Galileo was ...</td>
      <td>Copernicus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>ESPN's TOP 10 ALL-TIME ATHLETES</td>
      <td>$200</td>
      <td>No. 2: 1912 Olympian; football star at Carlisl...</td>
      <td>Jim Thorpe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EVERYBODY TALKS ABOUT IT...</td>
      <td>$200</td>
      <td>The city of Yuma in this state has a record av...</td>
      <td>Arizona</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>THE COMPANY LINE</td>
      <td>$200</td>
      <td>In 1963, live on "The Art Linkletter Show", th...</td>
      <td>McDonald's</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4680</td>
      <td>2004-12-31</td>
      <td>Jeopardy!</td>
      <td>EPITAPHS &amp; TRIBUTES</td>
      <td>$200</td>
      <td>Signer of the Dec. of Indep., framer of the Co...</td>
      <td>John Adams</td>
    </tr>
  </tbody>
</table>
</div>


Here's the full first row,

```python
data.values[0]
```
    array([4680, '2004-12-31', 'Jeopardy!', 'HISTORY', '$200',
           "For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory",
           'Copernicus'], dtype=object)

For our tf-idf matrix we'll need to know how many documents there are

```python
m = len(data)
```

# Building the TF-IDF matrix

```python
%%cython

import numpy as np
cimport numpy as np
import scipy.sparse as scs
from scipy.sparse import dok_matrix

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def unique_words(list sentences):
    cdef dict words = {}
    cdef int n = len(sentences)
    cdef int i, j
    for i in range(n):
        sent_list = [w.lower() for w in sentences[i].split(' ')]
        clean_sent_list = []
        for j in range(len(sent_list)):
            newword = ''
            for char in sent_list[j]:
                if char in alphabet:
                    newword += char
            clean_sent_list.append(newword)
        for word in clean_sent_list:
            if word != '':
                try:
                    words[word] += 1
                except KeyError:
                    words[word] = 1
    wordlist = sorted(words.keys())
    return wordlist, len(wordlist), words

# Use tf-idf
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
def populate_doc_matrix(docmatrix, wordlist, word_freq, np.ndarray data):
    cdef int n = len(data)   # number of documents
    cdef int i, j, k, m
    # construct word index first
    # This tells us (for any word) what index it is in in document
    print('Constructing Word Reference')
    wordref = {}
    for i in range(len(wordlist)):
        wordref[wordlist[i]] = i
    # Now populate sparse matrix
    print('Populating Sparse Matrix')
    for i in range(n):
        for j in range(2):
            words = [w.lower() for w in data[i, j].split(' ') if w != '']
            m = len(words)
            for k in range(m):
                word = words[k]
                cword = ''
                for char in word:
                    if char in alphabet:
                        cword += char
                if cword != '':
                    docmatrix[i, wordref[cword]] += 1
    # finish weighting
    print('Weighting Matrix')
    m, n = docmatrix.shape
    weighted_docmatrix = dok_matrix((m, n), dtype=float)
    for i in range(n):
        weighted_docmatrix[:, i] = docmatrix[:, i] * np.log(m / word_freq[wordlist[i]])
    return weighted_docmatrix, wordref
```

Building up some helper datasets

```python
words, n, wordfreq = unique_words(list(np.concatenate((data[[' Question']].values[:, 0],
                                  data[[' Answer']].values[:, 0]))))
```

Using these helper dictionaries we can track down the most frequent words,

```python
print('{} Documents (m) by {} Unique Words (n)\n\nTop 100 Most Frequent Words:{}'.format(
        m, n, ','.join([tup[0] for tup in sorted(wordfreq.items(), key=lambda tup: -tup[1])[:100]])))
```

    1000 Documents (m) by 5244 Unique Words (n)
    
    Top 100 Most Frequent Words:the,this,of,a,in,to,for,is,on,was,its,from,as,with,that,an,his,you,these,he,by,it,at,first,one,name,or,city,and,named,state,i,s,are,john,man,country,us,who,have,be,your,has,word,like,new,her,not,seen,called,when,hrefhttpwwwjarchivecommediadjjpg,had,out,were,here,about,can,clue,known,all,show,she,war,but,years,th,if,which,crew,make,now,film,made,wrote,series,may,type,island,more,used,area,than,began,queen,most,also,book,some,term,became,flag,said,part,river,youre,little,george,whose,him


And instead of storing this entire thing in memory we can use sparse matrix format

```python
docmatrix = dok_matrix((m, n), dtype=float)   # m-docs, n-unique words
ndocterm, wordref = populate_doc_matrix(docmatrix, words, wordfreq,
                                data[[' Question', ' Answer']].values)
```

    Constructing Word Reference
    Populating Sparse Matrix
    Weighting Matrix


Once we have our TF-IDF matrix, we perform SVD decomposition

```python
u, s, vt = ssl.svds(ndocterm.T, k=20)
u.shape, s.shape, vt.shape
```




    ((5244, 20), (20,), (20, 1000))




```python
np.save('umatrix.npy', u)
np.save('smatrix.npy', s)
np.save('vtmatrix.npy', vt)
```

Now that we have our $$k$$th-order decomposition, let's query the word "Species".


```python
wordref['species']
```
    4384

Document #4384 is most-related to the word "Species"

# Code

```python
#!/usr/bin/env python3


"""
Perform LSA on given set of text.

See README.md for details
"""


import sys
import re
import math
import time
import argparse
import concurrent.futures
import numpy as np
import scipy.sparse.linalg as ssl
from scipy.sparse import dok_matrix
from scipy.sparse import dok
from tqdm import tqdm
import numba
import enforce


def main():
    """ Manage Execution """
    args = get_args()

    with open(args.filename, 'r') as datafile:
        lines = datafile.read().split('\n')
        size = args.count
        if size == -1:
            size = len(lines)
        documents = np.empty(size, dtype=object)
        for i, line in enumerate(lines):
            if i >= size:
                break
            documents[i] = line

    doccount = len(documents)
    print('Program Start. Loaded Data. Time Elapsed: {}\n'.format(time.clock()))

    words = get_unique_words(documents, args.workers)
    wordcount = len(words.keys())
    topwords = ','.join([w for w, s in sorted(words.items(),
                                              key=lambda tup: -tup[1]['freq'])[:100]])

    print(('Found Word Frequencies\n'
           '\n{} Documents (m) by {} Unique Words (n)\n\n'
           'Top 100 Most Frequent Words:{}\n'
           'Time Elapsed: {}\n').format(doccount,
                                        wordcount,
                                        topwords,
                                        time.clock()))

    docmatrix = get_sparse_matrix(documents, words, args.workers)
    print('Calculated Sparse Matrix\nTime Elapsed: {}\n'.format(time.clock()))

    u, s, vt = ssl.svds(docmatrix, k=args.svdk)
    print('Calculated SVD Decomposition\nTime Elapsed: {}'.format(time.clock()))


@enforce.runtime_validation
def get_unique_words(documents: np.ndarray, workers: int) -> dict:
    """
    Parallelize Unique Word Calculation

    :documents: list of document strings
    :workers: number of workers

    :return: dictionary of word frequencies
    """
    data_bins = np.array_split(documents, workers)
    wordlist = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(unique_words, data_bins[i]):i for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Determining Unique Words', leave=True, total=workers):
            for word, stats in future.result().items():
                try:
                    wordlist[word]['freq'] += stats['freq']
                    wordlist[word]['doccount'] += stats['doccount']
                except KeyError:
                    wordlist[word] = {'freq':stats['freq'], 'doccount':stats['doccount']}
    return wordlist


@enforce.runtime_validation
def unique_words(data: np.ndarray) -> dict:
    """
    Finds unique word frequencies in documents

    :data: list of document strings

    :return: dictionary of word frequencies
    """
    words = {}
    olddoc = None
    for doc in data:
        for word in doc.split(' '):
            cword = re.sub('[^a-z]+', '', word.lower())
            if cword != '':
                try:
                    words[cword]['freq'] += 1
                    if doc != olddoc:
                        words[cword]['doccount'] += 1
                except KeyError:
                    words[cword] = {'freq':1, 'doccount':1}
        olddoc = doc
    return words


@enforce.runtime_validation
def get_sparse_matrix(documents: np.ndarray, words: dict, workers: int) -> dok.dok_matrix:
    """
    Parallelize Sparse Matrix Calculation

    :documents: list of document strings
    :words: dictionary of word frequencies
    :workers: number of workers

    :return: Sparse document term matrix
    """
    m = len(documents)
    n = len(words.keys())
    data_bins = np.array_split(documents, workers)
    docmatrix = dok_matrix((m, n), dtype=float)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(parse_docs, data_bins[i], words, len(documents)):i
                   for i in range(workers)}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Parsing Documents and Combining Arrays',
                           leave=True, total=workers):
            # THIS IS THE BOTTLENECK
            for key, value in future.result().items():
                docmatrix[key[0], key[1]] = value
    return docmatrix


@enforce.runtime_validation
def parse_docs(data: np.ndarray, words: dict, total_doc_count: int) -> dict:
    """
    Parallelize Sparse Matrix Calculation

    :data: list of document strings
    :words: dictionary of word frequencies
    :total_doc_count: total number of documents (for tf-idf)

    :return: Basically sparse array with weighted values
    """
    m = len(data)
    n = len(words.keys())
    docmatrix = {}
    wordref = {w:i for i, w in enumerate(sorted(words.keys()))}
    for i, doc in enumerate(data):
        for word in list(set([re.sub('[^a-z]+', '', w.lower()) for w in doc.split(' ')])):
            if word != '':
                docmatrix[(i, wordref[word])] = weight(total_doc_count,
                                                       words[word]['doccount'],
                                                       words[word]['freq'])
    return docmatrix


@numba.jit
def weight(total_doc_count: int, doccount: int, wordfreq: int) -> float:
    """
    Weighting function for Document Term Matrix.

    tf-idf => https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """
    return math.log(total_doc_count / doccount) * wordfreq


@enforce.runtime_validation
def get_args() -> argparse.Namespace:
    """
    Get Command line Arguments

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=32,
                        help=('Number of workers to use for multiprocessing'))
    parser.add_argument('-c', '--count', type=int, default=-1,
                        help=('Number of documents to use from original set'))
    parser.add_argument('-k', '--svdk', type=int, default=20,
                        help=('SVD Degree'))
    parser.add_argument('-f', '--filename', type=str, default='./data/jeopardy.csv',
                        help=('File to use for analysis'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
```
