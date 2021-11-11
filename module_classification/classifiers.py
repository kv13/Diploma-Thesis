import random
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import initializers
from sklearn.model_selection import StratifiedShuffleSplit

def my_dummy_classifier(tags,df_tags,issues_embeddings,cl_label,n_splits):
    
    target_label    = df_tags[cl_label]
    dummy_clf       = DummyClassifier(strategy = "uniform",random_state=0)
    total_confusion = np.zeros((2,2))
    
    # fit model 
    dummy_clf.fit(issues_embeddings,target_label)
    predictions = dummy_clf.predict(issues_embeddings)
    total_confusion = confusion_matrix(target_label,predictions)

    print(total_confusion)
    print("accuracy = TP+TN/(TP+TN+FP+FN)",(total_confusion[0][0]+total_confusion[1][1])/np.sum(total_confusion))
    print("geometric mean",np.sqrt((total_confusion[0][0]/(total_confusion[0][0]+total_confusion[0][1]))*
                                  (total_confusion[1][1]/(total_confusion[1][1]+total_confusion[1][0]))))
    print("\n")

def my_classifier(tags,df_tags,issues_embeddings,cl_label,n_splits):
    
    target_label    = df_tags[cl_label]
    counter_1       = np.sum(target_label)
    weight_0        = 1/(target_label.shape[0]-counter_1)
    weight_1        = 1/counter_1
    w               = {0:weight_0,1:weight_1}
    skf             = StratifiedKFold(n_splits)
    model           = LogisticRegression(solver='lbfgs',class_weight = w)
    total_confusion = np.zeros((2,2))
    counter         = 0
    auc             = 0
    for train_index, test_index in skf.split(issues_embeddings,target_label):
        
        X_train,X_test = issues_embeddings[train_index], issues_embeddings[test_index]
        y_train,y_test = target_label[train_index], target_label[test_index]
        
        #fit model 
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        
        #print(confusion_matrix(y_test,predictions))
        total_confusion = total_confusion+confusion_matrix(y_test,predictions)
        
        fpr,tpr,thresholds = metrics.roc_curve(y_test,model.predict_proba(X_test)[:,1])
        
        auc     = auc + metrics.auc(fpr,tpr)
        counter = counter +1
        
    print(total_confusion)
    print("accuracy = TP+TN/(TP+TN+FP+FN)",(total_confusion[0][0]+total_confusion[1][1])/np.sum(total_confusion))
    print("GM",np.sqrt((total_confusion[0][0]/(total_confusion[0][0]+total_confusion[0][1]))*
                                  (total_confusion[1][1]/(total_confusion[1][1]+total_confusion[1][0]))))
    print("Pre", total_confusion[0][0]/(total_confusion[0][1]+total_confusion[0][0]))
    print("AUC", auc/counter)
    print("\n")

def split_dataset2(issues_embeddings,target_labels,t_size =0.1):
    
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = t_size, random_state = 0)
    
    X_train_0 = list()
    X_train_1 = list()
    
    for train_index, test_index in sss.split(issues_embeddings,target_labels):
        #X_train,X_test = issues_embeddings[train_index], issues_embeddings[test_index]
        #Y_train,Y_test = target_labels[train_index], target_labels[test_index]
        
        
        X_test = issues_embeddings[test_index]
        Y_test = target_labels[test_index]
        
        for index in train_index:
            if target_labels.iloc[index] == 0:
                X_train_0.append(issues_embeddings[index])
            elif target_labels.iloc[index] == 1:
                X_train_1.append(issues_embeddings[index])
                
    return X_train_0,X_train_1,X_test,Y_test

def generate_batch(issues_embeddings,target_labels,batch_size):
    
    batch  = np.ndarray(shape = (batch_size,np.shape(issues_embeddings)[1]), dtype = np.float64)
    labels = np.ndarray(shape = (batch_size,2), dtype = np.float64)
    
    issues_to_use = random.sample([i for i in range(np.shape(issues_embeddings)[0])],batch_size)
    
    for counter,value in enumerate(issues_to_use):
        batch[counter][:]  = issues_embeddings[value][:]
        # label_0
        labels[counter][0] = 1-target_labels.iloc[value]
        # label_1
        labels[counter][1] =   target_labels.iloc[value]
    return batch,labels

def pooling(issues_embeddings_0, issues_embeddings_1, batch_size):
    
    batch  = np.ndarray(shape = (batch_size,np.shape(issues_embeddings_0)[1]), dtype = np.float64)
    labels = np.ndarray(shape = (batch_size,2), dtype = np.float64)
    
    issues_to_use_0 = random.sample([i for i in range(np.shape(issues_embeddings_0)[0])],batch_size//2)
    issues_to_use_1 = random.sample([i for i in range(np.shape(issues_embeddings_1)[0])],batch_size//2)
    
    # even indexes for issues belong to class 0
    # odd  indexes for issues belong to class 1
    counter_0 = 0
    counter_1 = 0
    
    for counter in range(batch_size):
        
        # even indexes
        if counter%2 == 0 :
            batch[counter][:]  = issues_embeddings_0[issues_to_use_0[counter_0]][:]
            labels[counter][0] = 1
            labels[counter][1] = 0
            counter_0 += 1
        else:
            batch[counter][:]  = issues_embeddings_1[issues_to_use_1[counter_1]][:]
            labels[counter][0] = 0
            labels[counter][1] = 1
            counter_1 += 1
            
    return batch,labels

def create_validation(train_issues_0,train_issues_1,rate=0.2):
    
    new_train_issues_0 = list()
    new_train_issues_1 = list()
    validation_issues  = list()
    validation_labels  = list()
    
    min_size = len(train_issues_0) if len(train_issues_0)<len(train_issues_1) else len(train_issues_1)
    print(min_size)
    validation_size_0 = int(min_size*rate)
    validation_size_1 = int(min_size*rate)
    
    validation_idxs_0 = random.sample([i for i in range(len(train_issues_0))], validation_size_0)
    validation_idxs_1 = random.sample([i for i in range(len(train_issues_1))], validation_size_1)
    
    for i in range(len(train_issues_0)):
        if i in validation_idxs_0:
            validation_issues.append(train_issues_0[i])
            validation_labels.append(0.0)
        else:
            new_train_issues_0.append(train_issues_0[i])
    
    for i in range(len(train_issues_1)):
        if i in validation_idxs_1:
            validation_issues.append(train_issues_1[i])
            validation_labels.append(1.0)
        else:
            new_train_issues_1.append(train_issues_1[i])
    
    # create a pandas series for validation labels in order to be compatible with the rest code
    val_labels_series = pd.Series(validation_labels, index = [i for i in range(len(validation_labels))])
    
    # create a np array for validation issues in order to be compatible with the rest code
    val_issues = np.array(validation_issues)
    
    return new_train_issues_0,new_train_issues_1,val_issues,val_labels_series

def compute_predictions(y_probs,v_labels):
    
    y_probs_1 = np.ndarray(shape = (np.shape(v_labels)[0],1), dtype = np.float64)
    y_preds_1 = np.ndarray(shape = (np.shape(v_labels)[0],1), dtype = np.float64)
    y_true_1  = np.ndarray(shape = (np.shape(v_labels)[0],1), dtype = np.float64) 
    
    for i in range(np.shape(v_labels)[0]):
        y_true_1[i]  = v_labels[i][1]
        y_preds_1[i] = 0 if y_probs[i][0]>y_probs[i][1] else 1
        y_probs_1[i] = y_probs[i][1]
    
    matrix_confusion = metrics.confusion_matrix(y_true=y_true_1,y_pred=y_preds_1)
    
    return y_probs_1, y_preds_1, y_true_1, matrix_confusion

def compute_auc(y_true,y_probs):
    
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_probs)
    auc                = metrics.auc(fpr,tpr)
    
    return auc

def compute_metrics(total_confusion,aucs):
    
    acc = (total_confusion[0][0]+total_confusion[1][1])/np.sum(total_confusion)
    
    gm  = np.sqrt((total_confusion[0][0]/(total_confusion[0][0]+total_confusion[0][1]))*
              (total_confusion[1][1]/(total_confusion[1][1]+total_confusion[1][0])))
    
    pre = total_confusion[1][1]/(total_confusion[1][1]+total_confusion[1][0])
    
    mean_auc = np.sum(aucs)/np.shape(aucs)[0]

    print("accuracy" , acc)
    print("precision", pre)
    print("GM"       , gm)
    print("mean auc" , mean_auc)
    print(total_confusion)
    print("\n")

def compute_predictions_voting(total_ypreds,total_nn):
    
    threshold = total_nn//2
    ypreds    = np.ndarray(shape = (np.shape(total_ypreds)[0],1),dtype = np.float64)
    
    for i in range(np.shape(total_ypreds)[0]):
        ypreds[i] = 0 if total_ypreds[i]<=threshold else 1
    
    return ypreds

def my_classifier_nn2(issues_embeddings_0,issues_embeddings_1,hidden_layer_dim,
                      learning_rate,batch_size,epochs,v_batch,v_labels):
    
    # input data
    X_train = tf.placeholder(tf.float64, shape=[None,np.shape(issues_embeddings_0)[1]])
    # input label
    Y_train = tf.placeholder(tf.float64, shape=[None,2])
    
    # input-hidden layer variables
    
    #initializer = initializers.GlorotNormal()
    #W1  = tf.Variable(initializer(shape=(np.shape(issues_embeddings_0)[1],hidden_layer_dim),dtype=tf.float64),name='W1')
    W1 = tf.Variable(tf.truncated_normal([np.shape(issues_embeddings_0)[1],hidden_layer_dim],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name='W1')
    
    b1 = tf.Variable(tf.random_normal([hidden_layer_dim],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name = 'b1')
    
    # hidden-output layer variables
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_dim,2],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim)
                                         ,dtype=tf.float64),name = 'W2')
    
    b2 = tf.Variable(tf.random_normal([2],dtype=tf.float64),name = 'b2')
    
    # neural network's functions
    hidden_layer   = tf.add(tf.matmul(X_train,W1),b1)
    hidden_layer   = tf.nn.tanh(hidden_layer)
     
    output_layer   = tf.add(tf.matmul(hidden_layer,W2),b2)
    output_layer_2 = tf.nn.softmax(output_layer)
    
    cost_func = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = Y_train,logits = output_layer))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_func)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(epochs):
            
            # generate batch.
            batch_x,batch_y = pooling(issues_embeddings_0,issues_embeddings_1,batch_size)
            
            # train the model
            _,loss = sess.run([optimizer,cost_func],feed_dict={X_train:batch_x,Y_train:batch_y})
        
        # saving the weights in numpy format
        #W1_np = W1.eval()
        #b1_np = b1.eval()
        #W2_np = W2.eval()
        #b2_np = b2.eval()
        
        
        # validation
        y_probs     = sess.run(output_layer_2,feed_dict={X_train:v_batch,Y_train:v_labels})
    
    return compute_predictions(y_probs,v_labels)

def my_classifier_nn3(issues_embeddings_0,issues_embeddings_1,hidden_layer_dim,
                      learning_rate,batch_size,v_batch,v_labels,t_batch,t_labels):
    
    # input data
    X_train = tf.placeholder(tf.float64, shape=[None,np.shape(issues_embeddings_0)[1]])
    # input label
    Y_train = tf.placeholder(tf.float64, shape=[None,2])
    
    # input-hidden layer variables
    W1 = tf.Variable(tf.truncated_normal([np.shape(issues_embeddings_0)[1],hidden_layer_dim],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name='W1')
    b1 = tf.Variable(tf.random_normal([hidden_layer_dim],stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                      dtype=tf.float64),name = 'b1')
    
    # hidden-output layer variables
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_dim,2],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name = 'W2')
    b2 = tf.Variable(tf.random_normal([2],dtype=tf.float64),name = 'b2')
    
    # neural network's functions
    hidden_layer   = tf.add(tf.matmul(X_train,W1),b1)
    hidden_layer   = tf.nn.tanh(hidden_layer)
     
    output_layer   = tf.add(tf.matmul(hidden_layer,W2),b2)
    output_layer_2 = tf.nn.softmax(output_layer)
    
    cost_func = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = Y_train,logits = output_layer))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_func)
    
    # patience mehtod's variables
    min_loss = float('inf')
    
    min_W1             = np.zeros((np.shape(issues_embeddings_0)[1],hidden_layer_dim))
    min_b1             = np.zeros(hidden_layer_dim)
    min_W2             = np.zeros((hidden_layer_dim,2))
    min_b2             = np.zeros(2)
    patience_remaining = 200
    step               = batch_size/(np.shape(issues_embeddings_0)[0] + np.shape(issues_embeddings_1)[0])
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(50000):
            
            # generate batch.
            batch_x,batch_y = pooling(issues_embeddings_0,issues_embeddings_1,batch_size)
            
            # train the model
            _,train_loss = sess.run([optimizer,cost_func],feed_dict={X_train:batch_x,Y_train:batch_y})
            # maybe valid loss should not be cross entropy but better the predictions.
            valid_loss   = sess.run(cost_func,feed_dict={X_train:v_batch,Y_train:v_labels})
            
            patience_remaining -= step
            if valid_loss < min_loss:
                min_loss           = valid_loss
                patience_remaining = 200
                min_W1             = W1.eval()
                min_b1             = b1.eval()
                min_W2             = W2.eval()
                min_b2             = b2.eval()
            if patience_remaining<=0:
                print("total epochs",epoch+1)
                break
        
        # restore minimum weights
        W1 = tf.convert_to_tensor(min_W1)
        b1 = tf.convert_to_tensor(min_b1)
        W2 = tf.convert_to_tensor(min_W2)
        b2 = tf.convert_to_tensor(min_b2)
                
        # testing
        y_probs     = sess.run(output_layer_2,feed_dict={X_train:t_batch,Y_train:t_labels})
        
    return compute_predictions(y_probs,t_labels)

def my_classifier_nn4(issues_embeddings_0,issues_embeddings_1,hidden_layer_dim,
                      learning_rate,batch_size,epochs,v_batch,v_labels):
    
    # input data
    X_train = tf.placeholder(tf.float64, shape=[None,np.shape(issues_embeddings_0)[1]])
    # input label
    Y_train = tf.placeholder(tf.float64, shape=[None,2])
    
    # input-hidden layer variables
    W1 = tf.Variable(tf.truncated_normal([np.shape(issues_embeddings_0)[1],hidden_layer_dim],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name='W1')
    b1 = tf.Variable(tf.random_normal([hidden_layer_dim],stddev = 1.0/ math.sqrt(hidden_layer_dim),dtype=tf.float64),name = 'b1')
    
    # hidden-output layer variables
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_dim,2],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name = 'W2')
    b2 = tf.Variable(tf.random_normal([2],dtype=tf.float64),name = 'b2')
    
    # neural network's functions
    hidden_layer   = tf.add(tf.matmul(X_train,W1),b1)
    hidden_layer   = tf.nn.tanh(hidden_layer)
    
    dropout_layer  = tf.nn.dropout(hidden_layer,rate = 0.5)
    
    output_layer   = tf.add(tf.matmul(dropout_layer,W2),b2)
    
    # for validation and testing dont use dropout
    output_layer_all = tf.add(tf.matmul(hidden_layer,W2),b2)
    output_layer_2   = tf.nn.softmax(output_layer_all)
    
    cost_func = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = Y_train,logits = output_layer))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_func)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(epochs):
            
            # generate batch.
            batch_x,batch_y = pooling(issues_embeddings_0,issues_embeddings_1,batch_size)
            
            # train the model
            _,loss = sess.run([optimizer,cost_func],feed_dict={X_train:batch_x,Y_train:batch_y})
        
        # validation
        y_probs     = sess.run(output_layer_2,feed_dict={X_train:v_batch,Y_train:v_labels})
    
    return compute_predictions(y_probs,v_labels)

def my_classifier_nn5(issues_embeddings_0,issues_embeddings_1,hidden_layer_dim,
                      learning_rate,batch_size,v_batch,v_labels,t_batch,t_labels):
    
    # input data
    X_train = tf.placeholder(tf.float64, shape=[None,np.shape(issues_embeddings_0)[1]])
    # input label
    Y_train = tf.placeholder(tf.float64, shape=[None,2])
    
    # input-hidden layer variables
    W1 = tf.Variable(tf.truncated_normal([np.shape(issues_embeddings_0)[1],hidden_layer_dim],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name='W1')
    b1 = tf.Variable(tf.random_normal([hidden_layer_dim],stddev = 1.0/ math.sqrt(hidden_layer_dim),dtype=tf.float64),name = 'b1')
    
    # hidden-output layer variables
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_dim,2],
                                         stddev = 1.0/ math.sqrt(hidden_layer_dim),
                                         dtype=tf.float64),name = 'W2')
    b2 = tf.Variable(tf.random_normal([2],dtype=tf.float64),name = 'b2')
    
    # neural network's functions
    hidden_layer   = tf.add(tf.matmul(X_train,W1),b1)
    hidden_layer   = tf.nn.tanh(hidden_layer)
    
    dropout_layer  = tf.nn.dropout(hidden_layer,rate = 0.5)
    
    output_layer   = tf.add(tf.matmul(dropout_layer,W2),b2)
    
    # for validation and testing dont use dropout
    output_layer_all = tf.add(tf.matmul(hidden_layer,W2),b2)
    output_layer_2   = tf.nn.softmax(output_layer_all)
    
    cost_func  = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = Y_train,logits = output_layer))
    valid_func = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = Y_train,logits = output_layer_all))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_func)
    
    # patience mehtod's variables
    min_loss = float('inf')
    
    min_W1             = np.zeros((np.shape(issues_embeddings_0)[1],hidden_layer_dim))
    min_b1             = np.zeros(hidden_layer_dim)
    min_W2             = np.zeros((hidden_layer_dim,2))
    min_b2             = np.zeros(2)
    
    patience_remaining = 100
    step               = batch_size/(np.shape(issues_embeddings_0)[0] + np.shape(issues_embeddings_1)[0])
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(50000):
            
            # generate batch.
            batch_x,batch_y = pooling(issues_embeddings_0,issues_embeddings_1,batch_size)
            
            # train the model
            _,loss     = sess.run([optimizer,cost_func],feed_dict={X_train:batch_x,Y_train:batch_y})
            valid_loss = sess.run(valid_func,feed_dict={X_train:v_batch,Y_train:v_labels}) 
            
            patience_remaining -= step
            if valid_loss < min_loss:
                min_loss           = valid_loss
                patience_remaining = 100
                min_W1             = W1.eval()
                min_b1             = b1.eval()
                min_W2             = W2.eval()
                min_b2             = b2.eval()
            if patience_remaining<=0:
                print("total epochs",epoch+1)
                break
        
        # restore minimum weights
        W1 = tf.convert_to_tensor(min_W1)
        b1 = tf.convert_to_tensor(min_b1)
        W2 = tf.convert_to_tensor(min_W2)
        b2 = tf.convert_to_tensor(min_b2)
        
        # testing
        y_probs     = sess.run(output_layer_2,feed_dict={X_train:t_batch,Y_train:t_labels})
    
    return compute_predictions(y_probs,t_labels)
