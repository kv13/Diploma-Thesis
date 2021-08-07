# import libraries.
import math
import time
import numpy as np
from sklearn import metrics
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers
from tensorflow.keras import initializers

# stack embeddings are created almost the same way 
# so many functions used in word embeddings will be used.
import module_embeddings.word_embeddings as we
import module_preprocessing.vocabulary_processes as vp

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
        embedding   = tf.Variable(initializer(shape=(vocabulary_size,embedding_dim)))
        #embedding = tf.Variable(tf.random_normal([vocabulary_size,embedding_dim]))
        
        # create the lookup table for each sample in X_train=>avoiding to use one_hot encoder
        X_embed   = tf.nn.embedding_lookup(embedding,X_train)
        
        # create variables for the loss function
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_dim],
                                                      stddev=1.0/ math.sqrt(embedding_dim)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
    
    loss_func = tf.reduce_sum(tf.nn.nce_loss(weights = nce_weights,biases =nce_biases,labels = Y_train,
                                              inputs = X_embed,num_sampled = num_sampled,
                                              num_classes = vocabulary_size ))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_opt = optimizer.minimize(loss_func)
    
    # Define initializer for tensorflow variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        # actual initialize the variables
        sess.run(init)
        
        # patience method's variables
        min_loss           = float('inf')
        min_emb_matrix     = np.zeros((vocabulary_size,embedding_dim))
        # patience step
        step               = skip_window*batch_size/len(corpus_indexes) 
        print("step is",step)
        patience_remaining = 100
        
        start_time = time.time()
        # train the model using 100 epoch patience
        for epoch in range(50000):
            
            # take a batch of data.
            batch_x,batch_y = we.generate_batch(corpus_data,corpus_indexes,batch_size)
            
            _,train_loss = sess.run([train_opt,loss_func],feed_dict={X_train:batch_x, Y_train:batch_y})
            valid_loss   = sess.run(loss_func,feed_dict={X_train:v_batch, Y_train:v_labels})
            
            patience_remaining     =patience_remaining - step
            if valid_loss < min_loss:
                min_loss           = valid_loss
                patience_remaining = 100
                min_emb_matrix     = embedding.eval()
            if patience_remaining <= 0:
                break
        
        # restore min embeddings
        embedding = tf.convert_to_tensor(min_emb_matrix)
        
        # normalize embeddings before using them
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding),1,keepdims = True))
        normalized_embedding = embedding/norm
        normalized_embedding_matrix = sess.run(normalized_embedding)
        
        #measure total time
        total_time = time.time() - start_time
        print("training time in seconds %s "%(str(total_time)))
        print("total epochs was",epoch+1)
        print("minimum loss is ",min_loss)
    
    return normalized_embedding_matrix,total_time,epoch+1


def stack_embeddings_creation(batch_size = 2048, skip_window=1,embedding_dim=8,num_sampled=64,learning_rate=0.1):

    mode = 'funcs'

    # load all important data
    corpus,corpus_indexes,vocabulary_size,v_dict,v_batch,v_label,testing_dict = we.load_all_data(mode)

    # actual training
    print("stack embeddings training starts with parameters vocabulary size",
            vocabulary_size," skip_window:",skip_window," embedding dim",
            embedding_dim," num sampled:",num_sampled," learning_rate:",
            learning_rate)
    
    embedding_matrix,m_time,m_epoch = model_def_cpu(corpus,corpus_indexes,batch_size,skip_window,embedding_dim,num_sampled,learning_rate,vocabulary_size,v_batch,v_label)

    # model validation
    v_auc = we.model_validation_v2(embedding_matrix,v_dict)
    print("Words embeddings model AUC on validation set:",v_auc)

    t_auc = we.model_validation_v2(embedding_matrix, testing_dict)
    print("Words embeddings model AUC on test set:",t_auc)

    # save embeddings matrix
    we.save_embeddings(mode,embedding_matrix)


def hyper_parameters_estimation(batch_size = 2048):

    # fixed parameters:
    unk_item    = "UNK"
    true_neigh  = 2
    false_neigh = 10
    valid_w     = 20
    valid_w2    = 40
    test_w      = 20
    
    # initialize variables
    b_t_auc         = float('-inf')
    b_v_auc         = float('-inf')
    b_min_occurance = -1
    b_skip_window   = -1
    b_embedding_dim = -1
    b_num_sampled   = -1
    b_learning_rate = -1

    # load data
    train_set      = vp.load_dataset('funcs', 'train.json')
    validation_set = vp.load_dataset('funcs', 'validation.json')
    test_set       = vp.load_dataset('funcs', 'test.json')

    # run estimations
    for min_occurance in [1,2,3,4,5]:
        
        # create vocabulary based on min_occurance parameter
        train_set_id,valid_set_id,test_set_id,vocab_dict = vp.create_vocabulary(train_set, validation_set, test_set, 'funcs', min_occurance, unk_item)
        vocabulary_size = len(vocab_dict)-1
        del vocab_dict

        for skip_window in [1,2,3,4]:
            # create corpus based on skip_window parameter
            corpus         = vp.create_corpus(train_set_id,skip_window)
            corpus_indexes = [w for w in range(len(corpus))]

            # compute testing and validation pairs based on test_set_id, valid_set_id
            _,test_dict    = vp.create_testing_dict(test_set_id , 1, test_w , 0       , true_neigh, false_neigh,skip_window=2)
            v_dict2,v_dict = vp.create_testing_dict(valid_set_id, 1, valid_w, valid_w2, true_neigh, false_neigh,skip_window=2)

            t_batch  = []
            t_label  = []
            for key in v_dict2:
                for value in v_dict2[key][0]:
                    t_batch.append(key)
                    t_label.append(value)
    
            v_batch = np.reshape(t_batch,(len(t_batch),))
            v_label = np.reshape(t_label,(len(t_label),1))

            for embedding_dim in [4,8,12,16]:
                for num_sampled in [16,32,64]:
                    for learning_rate in [0.01,0.1]:

                        # actual training
                        print("training starts with parameters vocabulary size", vocabulary_size," skip_window:", skip_window,
                        " min occurance", min_occurance," embedding dim", embedding_dim," num sampled:", num_sampled,
                        " learning_rate:", learning_rate)

                        # compute embeddings
                        embedding_matrix,m_time,m_epoch = model_def_cpu(corpus,corpus_indexes,batch_size,
                                                                        skip_window,embedding_dim,num_sampled,
                                                                        learning_rate,vocabulary_size,v_batch,v_label)

                        # compute auc
                        v_auc = we.model_validation_v2(embedding_matrix,v_dict)
                        t_auc = we.model_validation_v2(embedding_matrix,test_dict)

                        if v_auc > b_v_auc and t_auc > b_t_auc:
                            b_v_auc = v_auc
                            b_t_auc = t_auc
                            b_min_occurance = min_occurance
                            b_skip_window   = skip_window
                            b_embedding_dim = embedding_dim
                            b_num_sampled   = num_sampled
                            b_learning_rate = learning_rate

                        we.save_logs('funcs',m_epoch,m_time,v_auc,t_auc,min_occurance,skip_window,embedding_dim,num_sampled,learning_rate)
                time.sleep(60)
            time.sleep(60)
    
    print("best parameters: min_occurance",b_min_occurance," skip_window",b_skip_window," embedding_dim",b_embedding_dim,
          " num_sampled", b_num_sampled," learning_rate",b_learning_rate) 
    print("best validation AUC", b_v_auc)
    print("best testing AUC"   , b_t_auc)   
