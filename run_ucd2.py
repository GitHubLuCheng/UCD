#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Run the UCD model.

Usage: python run_ucd.py
Input data files: ../data/source_target.csv, ../data/user_friend_follower.csv, ../data/source_target.csv
"""

from __future__ import division, print_function

import os, pickle
import numpy as np
import pandas as pd

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Concatenate, Activation, Input
from keras.layers import Lambda, Embedding, GRU, Bidirectional, TimeDistributed, concatenate
from keras.models import Model
from keras import backend as K

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.preprocessing import normalize
from scipy import sparse

import tensorflow as tf
import networkx as nx

from modules.gae.cost import CostAE, CostVAE
from modules.gae.model import GCNModelAE, GCNModelVAE
from modules.gae.preprocessing_t import preprocess_graph, construct_feed_dict, sparse_to_tuple
from modules.gmm import GMM
from modules.attention_layer import AttLayer
from modules.estimation_net import EstimationNet

from util import Timer


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func)


def main():
    timer = Timer()
    timer.start()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tf.set_random_seed(0)

    MAX_SENT_LENGTH = 20
    MAX_SENTS = 100
    MAX_NB_WORDS = 2000
    EMBEDDING_DIM = 50
    POST_DIM = 10
    TEXT_DIM = 50
    TEST_SPLIT = 0.2

    mixtures = 5
    n_word = MAX_NB_WORDS
    Graph_DIM = 10
    training_epochs = 50

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', Graph_DIM, 'Number of units in hidden layer 2.')
    flags.DEFINE_integer('batch_size', 32, 'Size of a mini-batch')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('lambda1', 1e-4, 'Parameter of energy.')
    flags.DEFINE_float('lambda2', 1e-9, 'lossSigma.')
    flags.DEFINE_float('lambda3', 0.01, 'GAE.')
    flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
    model_str = FLAGS.model

    # variable to store evaluation results
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []

    for t in range(10):
        with open('./data/newdata.pickle', 'rb') as handle:
            store_data = pickle.load(handle, encoding='latin1')

        print(store_data.keys())
        # session label
        labels = store_data['labels']
        data = store_data['data']
        postInfo = store_data['postInfo']
        embedding_matrix = store_data['embedding_matrix']
        # word index for the whole corpus
        word_index = store_data['word_index']
        df = store_data['df']
        print(df.head(2))
        print(df.columns.tolist())
        num_session = len(df)
        num_test = int(TEST_SPLIT * num_session)
        timeInfo = store_data['timeInfo']
        # print(labels[:2])
        # print(data[:2])
        # print('shape', data[0].shape)
        # print(postInfo[:2])
        # print(embedding_matrix[:2])
        # # print(word_index)
        # print(nb_validation_samples)
        # print(df[:2])
        # print(timeInfo[:2])

        '''For Evaluation'''
        single_label = np.asarray(labels)
        labels = to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        zeros = np.zeros(num_session)
        zeros = zeros.reshape((num_session, 1, 1))
        #FLAGS.learning_rate = lr

        '''Hierarchical Attention Network for text and other info'''
        placeholders = {
            'zero_input': tf.placeholder(tf.float32, shape=[None, 1, 1]),
            'review_input': tf.placeholder(tf.float32, shape=[None, MAX_SENTS, MAX_SENT_LENGTH + 1]),
            'post_input': tf.placeholder(tf.float32, shape=[None, 4, ]),
            'time_label': tf.placeholder(tf.float32, shape=[None, MAX_SENTS])
        }

        # user following graph, source follows target
        user_following_graph = nx.Graph()
        with open('data/source_target.csv', 'r') as fin:
            fin.readline()
            for line in fin:
                src, tar = line.rstrip().split(',')
                user_following_graph.add_edge(src, tar)
        adj_user_following_graph = nx.adjacency_matrix(user_following_graph)

        user_popularity_pd = pd.read_csv('data/user_friend_follower.csv')
        user_popularity_dict = user_popularity_pd.set_index('user').T.to_dict('list')
        # {'surfer_magazine': [552490, 554], 'reneeyoungwwe': [257641, 499], ...}

        username_list = list(user_following_graph.nodes())
        num_user = len(username_list)
        user_session_mat = np.zeros((num_user, num_session))  # shape: num_user x num_session

        user_popularity_list = []
        for user_idx, username in enumerate(username_list):
            session_id_list = df.loc[df['owner_id'] == username].index.values
            for session_id in session_id_list:
                user_session_mat[user_idx, session_id] = 1
            user_popularity_list.append(user_popularity_dict[username])

        # split the train and test dataset
        user_session_train = user_session_mat[:, :-num_test]
        user_session_test = user_session_mat[:, -num_test:]

        user_popularity_list = sparse.csr_matrix(user_popularity_list)
        user_popularity_list = normalize(user_popularity_list, norm='max', axis=0)

        adj_norm = preprocess_graph(adj_user_following_graph)
        adj_label = adj_user_following_graph + sparse.eye(adj_user_following_graph.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Define placeholders
        placeholders.setdefault('features', tf.sparse_placeholder(tf.float32))
        placeholders.setdefault('adj', tf.sparse_placeholder(tf.float32))
        placeholders.setdefault('adj_orig', tf.sparse_placeholder(tf.float32))
        placeholders.setdefault('dropout', tf.placeholder_with_default(0., shape=()))
        placeholders.setdefault('user_post', tf.placeholder(tf.int32, [num_user, None]))
        d = {placeholders['dropout']: FLAGS.dropout}
        placeholders.update(d)
        num_nodes = adj_user_following_graph.shape[0]

        features = sparse_to_tuple(user_popularity_list.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        '''Graph AutoEncoder'''
        if model_str == 'gcn_ae':
            Graph_model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif model_str == 'gcn_vae':
            Graph_model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=True,
                                    mask_zero=True)

        all_input = Input(shape=(MAX_SENT_LENGTH + 1,))
        sentence_input = crop(1, 0, MAX_SENT_LENGTH)(all_input)  ##slice
        time_input = crop(1, MAX_SENT_LENGTH, MAX_SENT_LENGTH + 1)(all_input)  ##slice
        embedded_sequences = embedding_layer(sentence_input)
        #embedded_sequences=BatchNormalization()(embedded_sequences)
        l_lstm = Bidirectional(GRU(TEXT_DIM, return_sequences=True))(embedded_sequences)
        l_att = AttLayer(TEXT_DIM)(l_lstm)  ####(?,200)
        # time_embedding=Dense(TIME_DIM,activation='sigmoid')(time_input)
        merged_output = Concatenate()([l_att, time_input])  ###text+time information
        sentEncoder = Model(all_input, merged_output)

        review_input = placeholders['review_input']
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(TEXT_DIM, return_sequences=True))(review_encoder)
        fully_sent = Dense(1, use_bias=False)(l_lstm_sent)
        pred_time = Activation(activation='linear')(fully_sent)
        zero_input = placeholders['zero_input']
        shift_predtime = Concatenate(axis=1)([zero_input, pred_time])
        shift_predtime = crop(1, 0, MAX_SENTS)(shift_predtime)
        l_att_sent = AttLayer(TEXT_DIM)(l_lstm_sent)

        ###embed the #likes, shares
        post_input = placeholders['post_input']
        fully_post = Dense(POST_DIM, use_bias=False)(post_input)
        # norm_fullypost=BatchNormalization()(fully_post)
        post_embedding = Activation(activation='relu')(fully_post)
        fully_review = concatenate([l_att_sent,
                                    post_embedding])  ###merge the document level vectro with the additional embedded features such as #likes

        pos_weight = float(adj_user_following_graph.shape[0] * adj_user_following_graph.shape[0] - adj_user_following_graph.sum()) / adj_user_following_graph.sum()
        norm = adj_user_following_graph.shape[0] * adj_user_following_graph.shape[0] / float((adj_user_following_graph.shape[0] * adj_user_following_graph.shape[0] - adj_user_following_graph.sum()) * 2)
        with tf.name_scope('graph_cost'):
            preds_sub = Graph_model.reconstructions
            labels_sub = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                              validate_indices=False), [-1])
            if model_str == 'gcn_ae':
                opt = CostAE(preds=preds_sub, labels=labels_sub, pos_weight=pos_weight, norm=norm)
            elif model_str == 'gcn_vae':
                opt = CostVAE(preds=preds_sub,
                              labels=labels_sub,
                              model=Graph_model, num_nodes=num_nodes,
                              pos_weight=pos_weight,
                              norm=norm)
        User_latent = Graph_model.z_mean  ##(n_user, G_embeddim)
        Post_latent = fully_review  ###(batch size, text_embed_dim+post_dim)
        max_indices = tf.argmax(placeholders['user_post'], axis=0)
        add_latent = tf.gather(User_latent, max_indices)
        session_latent = tf.concat([Post_latent, add_latent], axis=1)  ###the representation of text + graph

        '''DAGMM'''
        h1_size = 2 * TEXT_DIM + Graph_DIM + POST_DIM
        gmm = GMM(mixtures)
        est_net = EstimationNet([h1_size, mixtures], tf.nn.tanh)
        gamma = est_net.inference(session_latent, FLAGS.dropout)
        gmm.fit(session_latent, gamma)
        individual_energy = gmm.energy(session_latent)

        Time_label = placeholders['time_label']
        Time_label = tf.reshape(Time_label, [tf.shape(Time_label)[0], MAX_SENTS, 1])

        with tf.name_scope('loss'):
            GAE_error = opt.cost
            energy = tf.reduce_mean(individual_energy)
            lossSigma = gmm.cov_diag_loss()
            prediction_error = tf.losses.mean_squared_error(shift_predtime, Time_label)
            loss = prediction_error + FLAGS.lambda1 * energy + FLAGS.lambda2 * lossSigma + FLAGS.lambda3 * GAE_error

        x_train = data[:-num_test]
        time_train = timeInfo[:-num_test]
        zeros_train = zeros[:-num_test]
        y_train = labels[:-num_test]
        post_train = postInfo[:-num_test]

        x_val = data[-num_test:]
        zeros_test = zeros[-num_test:]
        time_test = timeInfo[-num_test:]
        y_val = labels[-num_test:]

        post_test = postInfo[-num_test:]
        y_single = single_label[-num_test:]

        print('Number of positive and negative posts in training and validation set')
        print(y_train.sum(axis=0))
        print(y_val.sum(axis=0))
        print("model fitting - Unsupervised cyberbullying detection")

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_step = optimizer.minimize(loss)
        GAEcorrect_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                         tf.cast(labels_sub, tf.int32))
        feed_dict_train = construct_feed_dict(zeros_train, x_train, post_train, time_train, FLAGS.dropout, adj_norm,
                                              adj_label, features,
                                              user_session_train, placeholders)
        feed_dict_train.update({placeholders['dropout']: FLAGS.dropout})

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        total_batch = int(num_session / FLAGS.batch_size)
        zero_batches = np.array_split(zeros_train, total_batch)
        x_batches = np.array_split(x_train, total_batch)
        p_batches = np.array_split(post_train, total_batch)
        t_batches = np.array_split(time_train, total_batch)
        UP_batches = np.array_split(user_session_train, total_batch, axis=1)

        for epoch in range(training_epochs):
            ave_cost = 0
            ave_energy = 0
            ave_recon = 0
            ave_sigma = 0
            ave_GAE = 0
            for i in range(total_batch):
                batch_x = x_batches[i]
                batch_p = p_batches[i]
                batch_t = t_batches[i]
                batch_z = zero_batches[i]
                user_post = UP_batches[i]
                feed_dict = construct_feed_dict(batch_z, batch_x, batch_p, batch_t, FLAGS.dropout, adj_norm, adj_label,
                                                features, user_post,
                                                placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                _, total_loss, loss_sigma, GAE_loss, Energy_error, recon_error = sess.run(
                    [train_step, loss, lossSigma, GAE_error, energy, prediction_error], feed_dict)
                ave_cost += total_loss / total_batch
                ave_energy += Energy_error / total_batch
                ave_GAE += GAE_loss / total_batch
                ave_sigma += loss_sigma / total_batch
                ave_recon += recon_error / total_batch
            # if epoch % 10 == 0 or epoch == training_epochs - 1:
            # print (
            #          "This is epoch %d, the total loss is %f, energy error is %f, GAE error is %f, sigma error is %f,prediction error is %f") \
            #      % (epoch + 1, ave_cost, ave_energy, ave_GAE, ave_sigma, ave_recon)

        fix = gmm.fix_op()
        sess.run(fix, feed_dict=feed_dict_train)

        feed_dict_test = construct_feed_dict(zeros_test, x_val, post_test, time_test, FLAGS.dropout, adj_norm, adj_label,
                                             features, user_session_test, placeholders)
        pred_energy,representations = sess.run([individual_energy,session_latent], feed_dict=feed_dict_test)
        bully_energy_threshold = np.percentile(pred_energy, 65)
        print('the bully energy threshold is : %f' % bully_energy_threshold)
        label_pred = np.where(pred_energy >= bully_energy_threshold, 1, 0)
        print(precision_recall_fscore_support(y_single, label_pred))
        print(accuracy_score(y_single, label_pred))
        print(roc_auc_score(y_single, label_pred))
        tf.reset_default_graph()
        K.clear_session()

        precision_list.append(precision_recall_fscore_support(y_single, label_pred)[0][1])
        recall_list.append(precision_recall_fscore_support(y_single, label_pred)[1][1])
        f1_list.append(precision_recall_fscore_support(y_single, label_pred)[2][1])
        auc_list.append(roc_auc_score(y_single, label_pred))

    print('>>> Evaluation metrics')
    print('>>> precision mean: {0.4f}; precision std: {1:.4f}'.format(np.mean(precision_list), np.std(precision_list)))
    print('>>> recall mean: {0.4f}; recall std: {1:.4f}'.format(np.mean(recall_list), np.std(recall_list)))
    print('>>> f1 mean: {0.4f}; f1 std: {1:.4f}'.format(np.mean(f1_list), np.std(f1_list)))
    print('>>> auc mean: {0.4f}; auc std: {1:.4f}'.format(np.mean(auc_list), np.std(auc_list)))

    timer.stop()


if __name__ == '__main__':
    main()
