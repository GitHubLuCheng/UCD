#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Preprocess the Instagram dataset.

Usage: python preprocess.py
Input data files: ./instagram.csv
Output data files: ./instagram.pickle
"""

from __future__ import division, print_function

import sys, os, datetime, re, pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.utils import shuffle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ENG_STOPWORDS = stopwords.words('english')

MAX_SENT_LENGTH = 30
MAX_SENTS = 195
MAX_NB_WORDS = 2000
EMBEDDING_DIM = 50
DTFormat = '%Y-%m-%d %H:%M:%S'


def clean_html_tag(text):
    ret = ''
    to_write = True
    for char in text:
        if char == '<':
            to_write = False
        elif char == '>':
            to_write = True
        if to_write and char != '>':
            ret += char
    return ret


def clean_str(text):
    """ First lowercase, then remove English stopwords, then tokenize.
    """
    text = clean_html_tag(text.encode('ascii', 'ignore').decode('utf-8'))
    # lowercase
    text = text.strip().lower()
    # tokenization
    text = ' '.join([word for word in word_tokenize(text) if word not in ENG_STOPWORDS])
    return text


def main():
    input_filepath = './instagram.csv'
    output_filepath = './instagram.pickle'

    embeddings_index = {}
    with open('glove.twitter.27B.50d.txt', 'r', encoding='utf-8') as fin:
        for line in fin:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('total {0} word vectors'.format(len(embeddings_index)))

    df = pd.read_csv(input_filepath, encoding="ISO-8859-1", index_col=False)
    # fill nan cell
    df.fillna('', inplace=True)
    # shuffle the dataframe
    df = shuffle(df)
    print('data shape:', df.shape)

    labels = []
    comments_all = []  # all the comments
    social_content_all = []  # number of likes/shares/followed_by/follows for all the posts
    time_sequence_all = []  # time sequences for all the posts

    for _, session in df.iterrows():
        # number of likes/shares/followed_by/follows
        social_content = [int(re.findall(r'\d+', session['likes'])[0]), session['shared media'], session['followed_by'], session['follows']]
        social_content_all.append(social_content)

        label = session['question2']
        if label.startswith('n'):
            label = 0
        else:
            label = 1
        labels.append(label)

        post_time = ' '.join(session['cptn_time'].split()[-2:])  # datetime when the message is posted, %Y-%m-%d %H:%M:%S
        # handle corrupted time format -- some dates are missing the 2 at the head, length should be 19
        if len(post_time) == 18:
            post_time = '2' + post_time
        last_post_time = datetime.datetime.strptime(post_time, DTFormat)

        comments = []
        time_sequence = [0]  # time when the owner posts the picture

        for comment_idx in range(1, MAX_SENTS + 1):
            comment = session['clmn{0}'.format(comment_idx)]
            if comment.strip() and 'empety' not in comment:
                comment = comment.strip()
                identifier = '(created_at:'
                ts_start_idx = comment.find(identifier)
                if ts_start_idx != -1:
                    # comment timestamp
                    len_comment = len(comment)
                    ts = comment[ts_start_idx + len(identifier): len_comment - 1]
                    ts = datetime.datetime.strptime(ts, DTFormat)
                    time_lag = ts - last_post_time
                    time_sequence.append(time_lag.seconds)
                    last_post_time = ts

                    # comment text
                    comment = comment[: ts_start_idx]
                    comment = clean_str(comment)
                    comments.append(comment)

        comments_all.append(comments)
        time_sequence_all.append(time_sequence)

    pad_time_sequence_all = np.zeros((len(time_sequence_all), MAX_SENTS))
    for ts_idx, time_sequence in enumerate(time_sequence_all):
        pad_time_sequence_all[ts_idx][0: len(time_sequence)] = time_sequence
    # uniq_time_sequence_size = len(np.unique(pad_time_sequence_all))

    social_content_all = np.array(social_content_all)
    # uniq_social_content_size = len(np.unique(social_content_all))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    words_all = [' '.join(comments) for comments in comments_all]
    tokenizer.fit_on_texts(words_all)
    word_index = tokenizer.word_index

    text_tensor = np.zeros((len(comments_all), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    print('total {0:,} unique tokens'.format(len(word_index)))

    for session_idx, comments in enumerate(comments_all):
        for comment_idx, comment in enumerate(comments):
            if comment_idx < MAX_SENTS:
                word_idx = 0
                for _, word in enumerate(text_to_word_sequence(comment)):
                    if word_idx < MAX_SENT_LENGTH and word_index[word] < MAX_NB_WORDS:
                        text_tensor[session_idx, comment_idx, word_idx] = word_index[word]
                        word_idx += 1

    # Hierarchical Attention Network for text and other info
    pad_time_sequence_all = np.delete(pad_time_sequence_all, range(MAX_SENTS, pad_time_sequence_all.shape[1]), axis=1)
    pad_time_sequence_all = preprocessing.StandardScaler().fit_transform(pad_time_sequence_all)
    social_content_all = preprocessing.StandardScaler().fit_transform(social_content_all)
    print('text_tensor shape:', text_tensor.shape)
    print('pad_time_sequence_all shape:', pad_time_sequence_all.shape)
    print('social_content_all shape:', social_content_all.shape)

    han_data = np.dstack((text_tensor, pad_time_sequence_all))
    print('Hierarchical Attention Network data shape (text + time):', han_data.shape)

    embedding_matrix = np.random.random((len(word_index), EMBEDDING_DIM))
    for word, idx in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[idx] = embedding_vector

    store_data = {'embedding_matrix': embedding_matrix,
                  'data': han_data,
                  'labels': labels,
                  'postInfo': social_content_all,
                  'timeInfo': pad_time_sequence_all,
                  'word_index': word_index,
                  'df': df}

    with open(output_filepath, 'wb') as fout:
        pickle.dump(store_data, fout, protocol=pickle.HIGHEST_PROTOCOL)
    print('successfully write to output file {0}'.format(output_filepath))


if __name__ == '__main__':
    main()
