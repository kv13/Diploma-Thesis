import os
import time
import math
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


def save_logs(epoch,p_time,auc,t_auc,min_occurance,skip_window,embedding_dim,num_sampled,learning_rate):
    with open("outputs/logs.txt","a") as file:
        file.write("parameter's value: min occurance %s, skip window %s, embedding dim %s, num sampled %s, learning rate %s \n"%(str(min_occurance),str(skip_window),str(embedding_dim),str(num_sampled),str(learning_rate)))
        file.write("total time in sec %s and total epochs %s \n"%(str(p_time),str(epoch)))
        file.write("Validation AUC: %s \n"%(str(auc)))
        file.write("Testing AUC %s \n"%(str(t_auc)))


def save_embeddings(embedding_matrix):
    # make sure the directory exists
    directory = "results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = directory + '/' + 'word_embeddings.txt'
    np.savetxt(file_name,embedding_matrix,fmt = '%.8f')


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
    
    vocab_dict      = vp.load_vocabulary('words')
    vocabulary_size = len(vocab_dict)-1

    return corpus,corpus_indexes,vocabulary_size,v_dict,v_batch,v_label,testing_dict


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
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_dim],stddev=1.0 ))
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
        print("step is ",step)
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
        
    return normalized_embedding_matrix,total_time,epoch+1


def word_embeddings_creation(batch_size = 2048, skip_window=1, embedding_dim=64, num_sampled=64, learning_rate=0.1):

    # load all important data
    corpus,corpus_indexes,vocabulary_size,v_dict,v_batch,v_label,testing_dict = load_all_data()

    # actual training
    print("training starts with parameters vocabulary size",vocabulary_size," skip_window:",
            skip_window," embedding dim",embedding_dim," num sampled:",num_sampled," learning_rate:",learning_rate)

    embedding_matrix,m_time,m_epoch = model_def_cpu(corpus,corpus_indexes,batch_size,skip_window,
                                                    embedding_dim,num_sampled,learning_rate,vocabulary_size,v_batch,v_label)

    # model validation
    v_auc = model_validation_v2(embedding_matrix,v_dict)
    print("Words embeddings model AUC on validation set:",v_auc)

    t_auc = model_validation_v2(embedding_matrix, testing_dict)
    print("Words embeddings model AUC on test set:",t_auc)

    # save embeddings matrix
    save_embeddings(embedding_matrix)


def hyper_parameters_estimation(batch_size = 2048):
    
    # fixed parameters:
    unk_item    = "UNK"
    true_neigh  = 8
    false_neigh = 30
    valid_w     = 80
    valid_w2    = 70
    test_w      = 100

    # initialize variables
    b_t_auc         = float('-inf')
    b_v_auc         = float('-inf')
    b_min_occurance = -1
    b_skip_window   = -1
    b_embedding_dim = -1
    b_num_sampled   = -1
    b_learning_rate = -1

    # load data
    train_set      = vp.load_dataset('words', 'train.json')
    validation_set = vp.load_dataset('words', 'validation.json')
    test_set       = vp.load_dataset('words', 'test.json')

    # run estimations
    for min_occurance in [3,4,5,6,7,8,9,10]:

        # create vocabulary based on min_occurance parameter
        train_set_id,valid_set_id,test_set_id,vocab_dict = vp.create_vocabulary(train_set, validation_set, test_set, 'words', min_occurance, unk_item)
        vocabulary_size = len(vocab_dict)-1
        del vocab_dict

        for skip_window in [1,2,3,4,5]:
            # create corpus based on skip_window parameter
            corpus         = vp.create_corpus(train_set_id,skip_window)
            corpus_indexes = [w for w in range(len(corpus))] 
            
            # compute testing and validation pairs based on test_set_id, valid_set_id
            _,test_dict    = vp.create_testing_dict(test_set_id , 4, test_w , 0       , true_neigh, false_neigh)
            v_dict2,v_dict = vp.create_testing_dict(valid_set_id, 4, valid_w, valid_w2, true_neigh, false_neigh)

            t_batch  = []
            t_label  = []
            for key in v_dict2:
                for value in v_dict2[key][0]:
                    t_batch.append(key)
                    t_label.append(value)
    
            v_batch = np.reshape(t_batch,(len(t_batch),))
            v_label = np.reshape(t_label,(len(t_label),1))

            for embedding_dim in [32,64,128]:
                for num_sampled in [16,32,64]:
                    for learning_rate in [0.01,0.1]:
                        
                        # actual training
                        print("training starts with parameters vocabulary size", vocabulary_size," skip_window:", skip_window,
                        "min occurance", min_occurance," embedding dim", embedding_dim," num sampled:", num_sampled,
                        " learning_rate:", learning_rate)

                        # compute embeddings
                        embedding_matrix,m_time,m_epoch = model_def_cpu(corpus,corpus_indexes,batch_size,
                                                                        skip_window,embedding_dim,num_sampled,
                                                                        learning_rate,vocabulary_size,v_batch,v_label)
                        
                        # compute auc
                        v_auc = model_validation_v2(embedding_matrix,v_dict)
                        t_auc = model_validation_v2(embedding_matrix,test_dict)

                        if v_auc > b_v_auc and t_auc > b_t_auc:
                            b_v_auc = v_auc
                            b_t_auc = t_auc
                            b_min_occurance = min_occurance
                            b_skip_window   = skip_window
                            b_embedding_dim = embedding_dim
                            b_num_sampled   = num_sampled
                            b_learning_rate = learning_rate

                        save_logs(m_epoch,m_time,v_auc,t_auc,min_occurance,skip_window,embedding_dim,num_sampled,learning_rate)

                # sleep some time between neural network training to avoid overheating issues
                time.sleep(60)
            time.sleep(60)
    
    print("best parameters: min_occurance",b_min_occurance," skip_window",b_skip_window," embedding_dim",b_embedding_dim,
          " num_sampled", b_num_sampled," learning_rate",b_learning_rate) 
    print("best validation AUC", b_v_auc)
    print("best testing AUC"   , b_t_auc)                  


