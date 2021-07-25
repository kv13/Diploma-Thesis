import math
import time
import random
import numpy as np
from random import seed
from random import randint
from sklearn import metrics
from datetime import datetime
from module_preprocessing import vocabulary_processes as vp
from tensorflow.python.keras.backend import normalize_batch_in_training
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers
from tensorflow.keras import initializers


def load_all_data():
    corpus = vp.load_corpus('words')
    corpus_indexes = [w for w in range(len(corpus))] 

    # load validation sets
    v_dict  = vp.load_test_pairs('words','validation_pairs.pkl' )
    v_dict2 = vp.load_test_pairs('words','validation2_pairs.pkl')
    t_batch  = []
    t_label  = []
    for key in v_dict2:
        for value in v_dict2[key][0]:
            t_batch.append(key)
            t_label.append(value)

    v_batch = np.reshape(t_batch,(len(t_batch),))
    v_label = np.reshape(t_label,(len(t_label),1))

    # load testing set
    testing_dict = vp.load_test_pairs('words','testing_pairs.pkl')
    return corpus,corpus_indexes,v_dict,v_batch,v_label,testing_dict

def generate_batch(corpus_data,corpus_indexes,batch_size):
    
    batch  = np.ndarray(shape = (batch_size),   dtype = np.int32)
    labels = np.ndarray(shape = (batch_size,1), dtype = np.int32)
    
    seed(datetime.now())
    
    words_to_use = random.sample(corpus_indexes,batch_size)
    
    for counter,value in enumerate(words_to_use):
        batch[counter]    = corpus_data[value][0]
        labels[counter,0] = corpus_data[value][1] 
    
    return batch,labels


# The model computes tpr, fpr and auc. The classes are class_A = real neighbor
# and class_B = false neighbor. The model based on cosine similarity
# will try to predict the right label for each word pair given.
def model_validation_v2(embedding_matrix,words_dict):
    
    ylabels = list()
    ypreds  = list()
    
    for key in words_dict:
        target_emb = embedding_matrix[key]
        for true_neigh in words_dict[key][0]:
            neigh_emb = embedding_matrix[true_neigh]
            result    = np.dot(target_emb,neigh_emb)/(np.sqrt(np.dot(target_emb,target_emb))*np.sqrt(np.dot(neigh_emb,neigh_emb)))
            ylabels.append(1)
            ypreds.append(result)
            
        for false_neigh in words_dict[key][1]:
            neigh_emb = embedding_matrix[false_neigh]
            result    = np.dot(target_emb,neigh_emb)/(np.sqrt(np.dot(target_emb,target_emb))*np.sqrt(np.dot(neigh_emb,neigh_emb)))
            ylabels.append(0)
            ypreds.append(result)
    
    y = np.array(ylabels)
    score = np.array(ypreds)
    fpr,tpr,thresholds = metrics.roc_curve(y,score)
    auc = metrics.auc(fpr,tpr)
    return auc


def model_def_cpu(corpus_data,corpus_indexes,batch_size,skip_window,embedding_dim,
                  num_sampled,learning_rate,vocabulary_size,v_batch,v_labels):
    
    # Input data
    X_train = tf.placeholder(tf.int32, shape=[None])
    # Input label
    Y_train = tf.placeholder(tf.int32, shape=[None, 1])
    
    # ensure that the following ops & var are assigned to CPU
    with tf.device('/cpu:0'):
        
        # create the embedding variable wich contains the weights
        initializer = initializers.GlorotNormal()
        embedding = tf.Variable(initializer(shape=(vocabulary_size,embedding_dim)))
        
        # create the lookup table for each sample in X_train=>avoiding to use one_hot encoder
        X_embed   = tf.nn.embedding_lookup(embedding,X_train)
        
        # create variables for the loss function
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_dim],stddev=1.0))
        #nce_weights = tf.Variable(initializer(shape=(vocabulary_size,embedding_dim)))
        nce_biases  = tf.Variable(tf.zeros([vocabulary_size]))
    
    loss_func = tf.reduce_sum(tf.nn.nce_loss(weights = nce_weights,biases =nce_biases,labels = Y_train,
                                            inputs = X_embed,num_sampled = num_sampled,
                                            num_classes = vocabulary_size ))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_opt = optimizer.minimize(loss_func)
    
    #Define initializer for tensorflow variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        #actual initialize the variables
        sess.run(init)
        
        # patience method's variables 
        min_loss           = float('inf')
        min_emb_matrix     = np.zeros((vocabulary_size,embedding_dim))
        # patience step 
        step = 2*skip_window*batch_size/len(corpus_indexes)
        patience_remaining = 100
        
        start_time = time.time()
        # train the model using 100 epoch patience
        for epoch in range(50000):
            
            # take a batch of data.
            batch_x,batch_y = generate_batch(corpus_data,corpus_indexes,batch_size)
            
            _,train_loss = sess.run([train_opt,loss_func],feed_dict={X_train:batch_x, Y_train:batch_y})
            valid_loss   = sess.run(loss_func,feed_dict={X_train:v_batch, Y_train:v_labels})
            
            patience_remaining    -= step
            if valid_loss < min_loss:
                min_loss           = valid_loss
                patience_remaining = 100
                min_emb_matrix     = embedding.eval()
            if patience_remaining <= 0:
                break
        
        #restore min embeddings
        embedding = tf.convert_to_tensor(min_emb_matrix)
        
        #normalize embeddings before using them
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding),1,keepdims = True))
        normalized_embedding = embedding/norm
        normalized_embedding_matrix = sess.run(normalized_embedding)
        
        #measure total time
        total_time = time.time() - start_time
        print("training time in seconds %s "%(str(total_time)))
        print("total epochs was",epoch+1)
        
    return normalized_embedding_matrix


def word_embeddings_creation(batch_size = 2048, skip_window=1, embedding_dim=64, num_sampled=64, learning_rate=0.01):

    # load all important data
    corpus,corpus_indexes,v_dict,v_batch,v_label,testing_dict = load_all_data()

    # actual training    
    embedding_matrix = model_def_cpu(corpus,corpus_indexes,batch_size,skip_window,embedding_dim,num_sampled,learning_rate,v_batch,v_label)

    return embedding_matrix