import os
import string
import numpy as np
from itertools import islice
import random
import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import time
import json
import pandas as pd 
import pickle
import gzip
import struct
import numpy as np
from array import array
import glob 
import pickle
import gzip
#from PIL import Image 
#import scipy 
import random
import pdb

'''
    readcsv: Read feature tensors from csv data packet
    args:
        target: the directory that stores the csv files
        fealen: the length of feature tensor, related to to discarded DCT coefficients
    returns: (1) numpy array of feature tensors with shape: N x H x W x C
             (2) numpy array of labels with shape: N x 1 
'''

# get batch for testing

def batchfor_test(testX,testY,num,batch=1000):
    testlen=len(testX)
    if batch>testlen:
        print('ERROR:Batch size exceeds data size')
        print('Abort.')
    if num + batch < testlen:
        testX_batch = testX[num:num+batch]
        testY_batch = testY[num:num+batch]
        num = num + batch
    elif num + batch >= testlen:
        testX_batch = testX[num:testlen]
        testY_batch = testY[num:testlen]
        num = num +batch-testlen
    #testX_batch = np.reshape(testX_batch, [len(testX_batch),1,128*128])
    #testX_batch = np.rollaxis(testX_batch,1,3)
    test_batch = []
    test_batch.append(testX_batch)
    testY_batch = [vectorized_result(y) for y in testY_batch]
    test_batch.append(testY_batch)
    return test_batch, num

def vectorized_result(j):
    e = np.zeros(2)
    if j == 1:
        e[1] = 1
    else:
        e[0] = 1
    return e

#choose labeled data (from top, not randomly), just use for one time
def train_per(trainX,trainY,p):
    trainY = trainY.tolist()
    train_NHS = trainY.count(0)
    train_HS = trainY.count(1)
    print('trian_p = %f, train_NHS= %d, train_HS= %d' % (p,train_NHS,train_HS))
    num_HS = int(train_HS*p) +1
    num_NHS = int(train_NHS*p)
    num_train = num_HS + num_NHS
    train_nh_X, train_h_X = get_batch_nh_noweight(trainX,trainY)
    n_length = len(train_nh_X)
    h_length = len(train_h_X)
    #choose the first p of HS and NHS
    num_NHS_indx = list(range(0,num_NHS))
    num_HS_indx = list(range(0,num_HS))
    ft_batch = np.concatenate((train_nh_X[num_NHS_indx], train_h_X[num_HS_indx]))
    label = np.concatenate((np.zeros(num_NHS), np.ones(num_HS)))
    n_un_indx = list(range(num_NHS,train_NHS))
    h_un_indx = list(range(num_HS,train_HS))
    un_ft_batch = np.concatenate((train_nh_X[n_un_indx], train_h_X[h_un_indx]))
    un_label = np.concatenate((np.zeros(n_length-num_NHS), np.ones(h_length-num_HS)))
    print ("for train: NHS= %d, HS = %d, NHS+HS = %d,p = %f, unlabeled = %d" %(num_NHS,num_HS,len(label),p,len(un_label)))
    return ft_batch,label,un_ft_batch,un_label    




# choose labeled data based percentage with the same index from CC
def train_per_indexfromCNN(trainX,trainY,p,idxn,idxh):
    trainY = trainY.tolist()
    train_NHS = trainY.count(0)
    train_HS = trainY.count(1)
    print('trian_p = %f, train_NHS= %d, train_HS= %d' % (p,train_NHS,train_HS))
    num_HS = int(train_HS*p)+1
    num_NHS = int(train_NHS*p)
    num_train = num_HS + num_NHS
    train_nh_X, train_h_X = get_batch_nh_noweight(trainX,trainY)
    #n_length = len(train_nh_X)
    #h_length = len(train_h_X)
    #idxn =random.sample(list(range(0,n_length)),num_NHS)
    #idxh =random.sample(list(range(0,h_length)),num_HS)
    label = np.concatenate((np.zeros(num_NHS), np.ones(num_HS)))
        #label = processlabel(label,2, delta1, delta2)
    ft_batch = np.concatenate((train_nh_X[idxn], train_h_X[idxh]))
    n = list(range(0,train_NHS))
    h = list(range(0,train_HS))
    for i in idxn:
        n.remove(i)
    for j in idxh:
        h.remove(j)
    un_ft_batch = np.concatenate((train_nh_X[n], train_h_X[h]))
    un_label = np.concatenate((np.zeros(train_NHS-num_NHS), np.ones(train_HS-num_HS)))
    print ("for train: NHS= %d, HS = %d, NHS+HS = %d,p = %f, unlabeled = %d" %(len(idxn),len(idxh),len(label),p,len(un_label)))
    print("idxn =",idxn,"idxh =",idxh)
    return ft_batch,label,un_ft_batch,un_label    

#choose labeled data and get unlabeled data randomly(CNN)
def train_per_index(trainX,trainY,p):
    trainY = trainY.tolist()
    train_NHS = trainY.count(0)
    train_HS = trainY.count(1)
    print('trian_p = %f, train_NHS= %d, train_HS= %d' % (p,train_NHS,train_HS))
    num_HS = int(train_HS*p)+1
    num_NHS = int(train_NHS*p)
    num_train = num_HS + num_NHS
    train_nh_X, train_h_X = get_batch_nh_noweight(trainX,trainY)
    n_length = len(train_nh_X)
    h_length = len(train_h_X)
    idxn =random.sample(list(range(0,n_length)),num_NHS)
    idxh =random.sample(list(range(0,h_length)),num_HS)
    label = np.concatenate((np.zeros(num_NHS), np.ones(num_HS)))
        #label = processlabel(label,2, delta1, delta2)
    ft_batch = np.concatenate((train_nh_X[idxn], train_h_X[idxh]))
    n = list(range(0,train_NHS))
    h = list(range(0,train_HS))
    for i in idxn:
        n.remove(i)
    for j in idxh:
        h.remove(j)
    un_ft_batch = np.concatenate((train_nh_X[n], train_h_X[h]))
    un_label = np.concatenate((np.zeros(train_NHS-num_NHS), np.ones(train_HS-num_HS)))
    print ("for train: NHS= %d, HS = %d, NHS+HS = %d,p = %f, unlabeled = %d" %(num_NHS,num_HS,len(label),p,len(un_label)))
    print("idxn =",idxn,"idxh =",idxh, "n =",n, "h =",h)
    return ft_batch,label,un_ft_batch,un_label    

# get batch for training, with bias and weight
def get_batch_withweight_bias(trainX,trainY,weight,batchsize=32):
    train_nh_X, train_h_X, weight_nh,weight_h = get_batch_nh(trainX,trainY,weight)
    #batchsize = min(len(train_nh_X), len(train_h_X), batchsize)
    n_length = len(train_nh_X)
    h_length = len(train_h_X)
    if h_length < 16:
        if not h_length ==0:
            idxn = (np.random.rand(batchsize-h_length)*n_length).astype(int)
            ft_batch = np.concatenate((train_nh_X[idxn],train_h_X))
            label = np.concatenate((np.zeros(batchsize-h_length), np.ones(h_length)))
            weight_whole = np.concatenate((weight_nh[idxn],weight_h))
            ft_batch_nhs = train_nh_X[idxn]
            label_nhs = np.zeros(batchsize-h_length)
            weight_nhs = weight_nh[idxn]
        else:
            idxn = (np.random.rand(batchsize)*n_length).astype(int)
            ft_batch = train_nh_X[idxn]
            label = np.zeros(batchsize)
            weight_whole = weight_nh[idxn]
            ft_batch_nhs = ft_batch
            label_nhs = label
            weight_nhs = weight_whole
    else:
        num = batchsize/2
        idxn =(np.random.rand(num)*n_length).astype(int)
        idxh = (np.random.rand(num)*h_length).astype(int)
        ft_batch = np.concatenate((train_nh_X[idxn], train_h_X[idxh]))
        label = np.concatenate((np.zeros(num), np.ones(num)))
        weight_whole = np.concatenate((weight_nh[idxn],weight_h[idxh]))
        ft_batch_nhs = train_nh_X[idxn]
        label_nhs = np.zeros(num)
        weight_nhs = weight_nh[idxn]
    #print(ft_batch.shape, ft_batch_nhs.shape)    
    ft_batch = np.reshape(ft_batch,[batchsize,1,128*128])
    ft_batch = np.rollaxis(ft_batch,1,3)
    ft_batch_nhs = np.reshape(ft_batch_nhs,[len(ft_batch_nhs),1,128*128])
    ft_batch_nhs = np.rollaxis(ft_batch_nhs,1,3)
    return ft_batch, label, weight_whole, ft_batch_nhs, label_nhs, weight_nhs


# get batch for training(cancle the nhs part for delta)
def get_batch_withweight(trainX,trainY,weight,batchsize=32):
    train_nh_X, train_h_X, weight_nh,weight_h = get_batch_nh(trainX,trainY,weight)
    #batchsize = min(len(train_nh_X), len(train_h_X), batchsize)
    n_length = len(train_nh_X)
    h_length = len(train_h_X)
    if h_length < 16:
        if not h_length ==0:
            idxn = (np.random.rand(batchsize-h_length)*n_length).astype(int)
            ft_batch = np.concatenate((train_nh_X[idxn],train_h_X))
            label = np.concatenate((np.zeros(batchsize-h_length), np.ones(h_length)))
            weight_whole = np.concatenate((weight_nh[idxn],weight_h))
        else:
            idxn = (np.random.rand(batchsize)*n_length).astype(int)
            ft_batch = train_nh_X[idxn]
            label = np.zeros(batchsize)
            weight_whole = weight_nh[idxn]
    else:
        num = batchsize/2
        idxn =(np.random.rand(num)*n_length).astype(int)
        idxh = (np.random.rand(num)*h_length).astype(int)
        ft_batch = np.concatenate((train_nh_X[idxn], train_h_X[idxh]))
        label = np.concatenate((np.zeros(num), np.ones(num)))
        weight_whole = np.concatenate((weight_nh[idxn],weight_h[idxh]))
    #print(ft_batch.shape, ft_batch_nhs.shape)    
    #ft_batch = np.reshape(ft_batch,[batchsize,1,128*128])
    #ft_batch = np.rollaxis(ft_batch,1,3)
    return ft_batch, label, weight_whole

def get_batch_nh(trainX,trainY,weight):
    trainwhole = list(zip(trainX,trainY,weight))
    train_nh_X = []
    train_h_X = []
    weight_nh = []
    weight_h = []
    for i in trainwhole:
        x,y,z = i
        if y == 0:
            train_nh_X.append(x)
            weight_nh.append(z)
        if y == 1:
            train_h_X.append(x)
            weight_h.append(z)
    #train_nh_Y = np.zeros(len(train_nh_X))
    train_nh_X = np.array(train_nh_X)
    train_h_X = np.array(train_h_X)
    weight_nh = np.array(weight_nh)
    weight_h = np.array(weight_h)
    return(train_nh_X, train_h_X,weight_nh,weight_h)

def get_batch_nh_noweight(trainX,trainY):
    trainwhole = list(zip(trainX,trainY))
    train_nh_X = []
    train_h_X = []
    for i in trainwhole:
        x,y = i
        if y == 0:
            train_nh_X.append(x)
        if y == 1:
            train_h_X.append(x)
    #train_nh_Y = np.zeros(len(train_nh_X))
    train_nh_X = np.array(train_nh_X)
    train_h_X = np.array(train_h_X)
    return(train_nh_X, train_h_X)
#get Dkl
def get_Dkl(x,P):
    N = tf.shape(x)[0]
    x_N = tf.reshape(tf.tile(x, [N,1]), [N,N,2])
    x_NT = tf.transpose(x_N, perm=[1,0,2])
    D1 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(x_NT),logits=x_N)
    #D1 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(x_NT),logits=(x_NT/x_N))
    D = tf.negative(D1)
    P_D = tf.ones([1,N])[0]
    P_D_diag = tf.diag(P_D)
    Dii_diag = D*P_D_diag
    Dii_1 = tf.reduce_sum(Dii_diag,1)
    P_D_1 = tf.ones_like(D)
    Dii = Dii_1 * P_D_1
    D_kl = Dii - D
    D_kl1 = tf.multiply(D_kl,P)
    P1 = tf.cast(tf.ones_like(P), tf.float32)
    P2 = 2*P1
    P0 = tf.cast((P1 - P),tf.float32)
    D_kl0 = tf.multiply(D_kl,P0)
    D_kl_C = P2 - D_kl0
    P_sign = tf.sign(D_kl_C)
    P_sign_2 = P_sign + P1
    P_sign_1 = tf.div(P_sign_2, 2)
    C0 = tf.multiply(D_kl_C, P_sign_1)
    C0_t = tf.multiply(C0,P0)
    C_whole = D_kl1 + C0_t
    C_loss1 = tf.reduce_mean(C_whole)
    C_loss = (C_loss1)/2
    return D_kl,C_loss

#softmax_cross_entropy, for Dkl,no-
def softmax_cross_C(labels, logits):
    logits_s  = np.log(logits)
    loss = labels * logits_s
    loss_whole = loss.sum(2)
    return(loss_whole)
    
def softmax_np(x):
    x_exp = np.exp(x)
    x_exp_whole = x_exp.sum(1)
    for i in range(len(x)):
        x_exp[i]=x_exp[i]/x_exp_whole[i]
    return x_exp

def sigmoid_np(x):
    x_nexp = np.exp(np.negative(x))
    Pone = np.ones_like(x)
    y = 1/(Pone + x_nexp)
    return y
    
def get_Dkl_C(x,P):
    N = len(x)
    x = softmax_np(x)
    x_N = (np.tile(x, [N,1])).reshape(N,N,2)
    x_NT = np.transpose(x_N,[1,0,2])
    D = softmax_cross_C(x_NT, x_N)
    #D1 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(x_NT),logits=x_N)
    #D1 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(x_NT),logits=(x_NT/x_N))
    #D = np.negative(D1)
    P_D = np.ones(N)
    P_D_diag = np.diag(P_D)
    Dii_diag = D*P_D_diag
    Dii_1 = np.sum(Dii_diag,1)
    P_D_1 = np.ones_like(D)
    Dii = Dii_1 * P_D_1
    D_kl = Dii - D
    D_kl1 = np.multiply(D_kl,P)
    P1 = np.ones_like(P)
    P1 = P1.astype(np.float32)
    P2 = 2*P1
    P0 = (P1 - P).astype(np.float32)
    D_kl0 = np.multiply(D_kl,P0)
    D_kl_C = P2 - D_kl0
    
    P_sign = np.sign(D_kl_C)
    P_sign_2 = P_sign + P1
    P_sign_1 = P_sign_2/2
    C0 = np.multiply(D_kl_C, P_sign_1)
    C0_t = np.multiply(C0,P0)
    C_whole = D_kl1 + C0_t
    C_loss1 = np.mean(C_whole)
    C_loss = (C_loss1)/2
    return D_kl,C_loss

#get wi,CPU
def get_wi(D_kl,P):
    P_whole = (np.sum(P,1)).astype(np.float32)
    D_kli1 = np.multiply(D_kl,P)
    D_kli = np.sum(D_kli1, 1)
    di = D_kli/P_whole
    di_exp = np.exp(np.negative(di))
    di_exp_whole = np.sum(di_exp)
    len_en = len(di)
    #len_en = len_en1.astype(np.float32)
    wi = di_exp/di_exp_whole*len_en#weight   
    #wi = sigmoid_np(wi_1)
    return wi

#get pair contraint
def pairwise_constraint(trainX, trainY):
        plen = len(trainX)
        #Ylen = len(trainY[0])
        pconstrain = np.zeros([plen,plen])
        for i in range(plen):
            for j in range(plen):
                if trainY[i] == trainY[j]:
                    pconstrain[i][j] = 1
        return pconstrain

# data shape
def data_flatten(predict_label_whole,predict_label):
    #predict_label_whole=[]
    predict1 = predict_label.flatten()
    predict2 = predict1.tolist()
    predict_label_whole = predict_label_whole + predict2
    return predict_label_whole
def data_flatten2(predict_label_whole):
    predict_label2 = np.array(predict_label_whole)
    predict_label1 = predict_label2.flatten()
    return predict_label1

# get unX,unY,weight for defining r
def get_data_un_r(un_for_r):
    X=[]
    Y=[]
    W=[]
    for i in un_for_r:
        x,y,z,h = i
        X.append(x)
        Y.append(y)
        W.append(z)
    return X,Y,W

# split X and Y
def data_split(newdata):
    dataX = []
    dataY = []
    for i in newdata:
        x,y = i
        dataX.append(x)
        dataY.append(y)
    return dataX,dataY

def data_split_sencond(newdata):
    dataX = []
    dataY = []
    for i in range(len(newdata)):
        if i % 2 ==0:
            dataX.append(newdata[i])
        if i % 2 ==1:
            dataY.append(newdata[i])
    return(dataX,dataY)  
    
'''
    readcsv: Read feature tensors from csv data packet
    args:
        target: the directory that stores the csv files
        fealen: the length of feature tensor, related to to discarded DCT coefficients
    returns: (1) numpy array of feature tensors with shape: N x H x W x C
             (2) numpy array of labels with shape: N x 1 
'''
def readcsv(target, maxlen, ratio, fealen=32):
    #read label
    path  = target + '/label.csv'
    label_all = np.genfromtxt(path, delimiter=',')
    #read feature
    feature_all = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in xrange(0, len(filenames)-1):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature_all.append(featemp)
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature_all.append(featemp)          
    for i in range(len(feature_all)):
        print(feature_all[i].shape)
    # Yibo: adjust amount of training data 
    # get labeled data and unlabeled data
    #pdb.set_trace()
    feature_all = np.rollaxis(np.asarray(feature_all), 0, 3)[:,:,0:fealen]
    if not ratio == 1:
        feature = feature_all[:maxlen]
        label = label_all[:maxlen]
        feature_un = feature_all[maxlen:]
        label_un = label_all[maxle:]    
        return feature, label, feature_un, label_un
    else:
        return feature_all, label_all
'''
    processlabel: adjust ground truth for biased learning
    args:
        label: numpy array contains labels
        cato : number of classes in the task
        delta1: bias for class 1
        delta2: bias for class 2
    return: softmax label with bias
'''
def processlabel(label, cato=2, delta1 = 0, delta2=0):
    softmaxlabel=np.zeros(len(label)*cato, dtype=np.float32).reshape(len(label), cato)
    for i in range(0, len(label)):
        if int(label[i])==0:
            softmaxlabel[i,0]=1-delta1
            softmaxlabel[i,1]=delta1
        if int(label[i])==1:
            softmaxlabel[i,0]=delta2
            softmaxlabel[i,1]=1-delta2
    return softmaxlabel
'''
    loss_to_bias: calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
'''
def loss_to_bias(loss,  alpha, threshold=0.3):
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1+np.exp(alpha*loss))
    return bias

'''
    forward: define the neural network architecute
    args:
        input: feature tensor batch with size B x H x W x C
        is_training: whether the forward process is training, affect dropout layer
        reuse: undetermined
        scope: undetermined
    return: prediction socre(s) of input batch
'''
def forward(input, is_training=True, reuse=False, scope='model', flip=False):
    if flip == True:
        input = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input)
        input = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), input)

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(input, 16, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
            net = slim.flatten(net)
            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
    return predict

'''
    data: a class to handle the training and testing data, implement minibatch fetch
    args: 
        fea: feature tensor of whole data set
        lab: labels of whole data set
        ptr: a pointer for the current location of minibatch
        maxlen: length of entire dataset
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
    methods:
        nextinstance():  returns a single instance and its label from the training set, used for SGD
        nextbatch(): returns a batch of instances and their labels from the training set, used for MGD
            args: 
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label
        sgd_batch(): returns a batch of instances and their labels from the trainin set randomly, number of hs and nhs are equal.
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label

'''
class data:
    def __init__(self, fea, lab, preload=False, ratio=None):
        self.ptr_n=0
        self.ptr_h=0
        self.ptr=0
        self.dat=fea
        self.label=lab
        with open(lab) as f:
            self.maxlen=sum(1 for _ in f)
            # Yibo
            if ratio:
                content = "adjust data from %d " % (self.maxlen)
                self.maxlen = int(self.maxlen*min(ratio, 1.0))
                content += " to %d (%g)" % (self.maxlen, ratio)
                print(content)
        if preload:
            print("loading data into the main memory...")
            # Yibo
            self.ft_buffer, self.label_buffer, self.ft_buffer_un, self.label_buffer_un=readcsv(self.dat, self.maxlen)

    def nextinstance(self):
        temp_fea=[]
        label=None
        idx=random.randint(0,self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))        
        with open(self.label) as l:
            temp_label=np.asarray(list(l)[idx]).astype(int)
            if temp_label==0:
                label=[1,0]
            else:
                label=[0,1]
        return np.rollaxis(np.array(temp_fea),0,3),np.array([label])

    def sgd(self, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            # Yibo
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
        length=labelist.size
        idx=random.randint(0, length-1)
        temp_label=labelist[idx]
        if temp_label==0:
            label=[1,0]
        else:
            label=[0,1]
        ft= self.ft_buffer[idx]

        return ft, np.array(label)
    def sgd_batch_2(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            # Yibo
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch / 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        label = processlabel(label,2, 0,0 )
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label


    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            # Yibo
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch / 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        #label = processlabel(label,2, delta1, delta2)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_beta: returns the balalced batch, used for training only
    '''
    def nextbatch_beta(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        with open(self.label) as l:
            labelist=np.asarray(list(l)[:self.maxlen]).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size

        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch/2
            # Yibo: handle small data size 
            # I change num for non-hotspot to batch-num 
            if num >= h_length:
                num = h_length-1
            if num>=n_length or num>=h_length:
                print('ERROR:Batch size exceeds data size')
                print('Abort.')
                quit()
            else:
                if self.ptr_n+(batch-num) <n_length:
                    idxn = labexn[self.ptr_n:self.ptr_n+(batch-num)]
                elif self.ptr_n+(batch-num) >=n_length:
                    idxn = np.concatenate((labexn[self.ptr_n:n_length], labexn[0:self.ptr_n+(batch-num)-n_length]))
                self.ptr_n = update_ptr(self.ptr_n, (batch-num), n_length)
                if self.ptr_h+num <h_length:
                    idxh = labexh[self.ptr_h:self.ptr_h+num]
                elif self.ptr_h+num >=h_length:
                    idxh = np.concatenate((labexh[self.ptr_h:h_length], labexh[0:self.ptr_h+num-h_length]))
                self.ptr_h = update_ptr(self.ptr_h, num, h_length)
                #print self.ptr_n, self.ptr_h
                label = np.concatenate((np.zeros(batch-num), np.ones(num)))
                #label = processlabel(label,2, delta1, delta2)
                ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
                ft_batch_nhs = self.ft_buffer[idxn]
                label_nhs = np.zeros(batch-num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''
    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr+batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr+batch]
        else:
            label = np.concatenate((self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr+batch-self.maxlen]))
            ft_batch = np.concatenate((self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr+batch-self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch, label
    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        #print('recommed to use nextbatch_beta() instead')
        databat=None
        temp_fea=[]
        label=None
        if batch>self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr+batch < self.maxlen:
            #processing labels
            with open(self.label) as l:
                temp_label=np.asarray(list(l)[self.ptr:self.ptr+batch])
                label=processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
            self.ptr=self.ptr+batch
        elif (self.ptr+batch) >= self.maxlen:
            
            #processing labels
            with open(self.label) as l:
                a=np.genfromtxt(islice(l, self.ptr, self.maxlen),delimiter=',')
            with open(self.label) as l:
                b=np.genfromtxt(islice(l, 0, self.ptr+batch-self.maxlen),delimiter=',')
            #processing data
            if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                temp_label=b
            elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                temp_label=a
            else:
                temp_label=np.concatenate((a,b))
            label=processlabel(temp_label,2, delta1, delta2)
            #print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, None, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print a.shape, b.shape, self.ptr
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, 0, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print a.shape, b.shape, self.ptr
            self.ptr=self.ptr+batch-self.maxlen
        #print np.asarray(temp_fea).shape
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:,:,0:channel], label

def DCT_data(self, fea, lab, preload=False, ratio=None):
    self.ptr_n=0
    self.ptr_h=0
    self.ptr=0
    self.dat=fea
    self.label=lab
    with open(lab) as f:
        self.maxlen=sum(1 for _ in f)
        # Yibo
        if ratio:
            content = "adjust data from %d " % (self.maxlen)
            self.maxlen = int(self.maxlen*min(ratio, 1.0))
            content += " to %d (%g)" % (self.maxlen, ratio)
            print(content)
    if preload:
        print("loading data into the main memory...")
        # Yibo
        # get data according to ratio
        if not ratio == 1:    
            self.ft_buffer, self.label_buffer, self.ft_buffer_un, self.label_buffer_un=readcsv(self.dat, self.maxlen, ratio)
            return(self.ft_buffer, self.label_buffer, self.ft_buffer_un, self.label_buffer_un)
        else:
            self.ft_buffer, self.label_buffer=readcsv(self.dat, self.maxlen, ratio)
            return(self.ft_buffer, self.label_buffer)


class SampleData (object): 
    """
    initialization
    """
    def __init__(self):
        self.trainDataX = []#np.empty((0))  # train data 
        self.trainDataY =[] #np.empty((0))  # train data 
        self.testDataX = [] #np.empty((0)) # test data, actually including train data in the front as well  
        self.testDataY = [] #np.empty((0)) # test data 
        self.testDataPredictY = [] #np.empty((0)) # predicted y for test data 
    def load_data(self, file1,flip = 0, numTrainRatio = 1, duplicateFactor = 0):
        #f1 = gzip.open('trainset_1024x1024_packed.pklz', 'rb') 
        #training_data= pickle.load(f1, encoding='latin1')
        #f2 = gzip.open('testset_1024x1024_packed.pklz', 'rb') 
        #test_data= pickle.load(f2, encoding='latin1')
        #f1.close()
        #f2.close()
        #return(training_data)
        tt = time()
        print("reading %s" % (file1))
        with gzip.open(file1, "rb") as f:
                    dataX, dataY = pickle.load(f)
                    print("%d samples" % (len(dataX)))
                    self.trainDataX.extend(dataX)
                    self.trainDataY.extend(dataY)
                          
        #flip hotspots for balancing
        if  flip == 1:
            numTrainSamples = int(len(self.trainDataX)*numTrainRatio)
            for i in range(numTrainSamples):
                if self.trainDataY[i]: #hotspot
                    image = np.unpackbits(self.trainDataX[i]).reshape([128,128])
                    label = self.trainDataY[i]
                    self.trainDataX.append(np.packbits(self.fliplr(image)))
                    self.trainDataY.append(label)
                    self.trainDataX.append(np.packbits(self.flipud(image)))
                    self.trainDataY.append(label)
                    self.trainDataX.append(np.packbits(self.flipud(self.fliplr(image))))
                    self.trainDataY.append(label)
            # update after flipping
            numTrainSamples = int(len(self.trainDataX)*numTrainRatio)
            print("after flipping #training = %d, #positive = %d" % (numTrainSamples, self.trainDataY[:numTrainSamples].count(1)))
            
        # duplicate hotspots for balancing
        if duplicateFactor > 1:
            print("duplicate hotspots for balancing")
            numTrainSamples = int(len(self.trainDataX)*numTrainRatio)
            for i in range(numTrainSamples):
                if self.trainDataY[i]: # hotspot 
                    image = self.trainDataX[i]
                    label = self.trainDataY[i]
                    for j in range(1, int(duplicateFactor)):
                        self.trainDataX.append(image)
                        self.trainDataY.append(label)
            # update after flipping 
            numTrainSamples = int(len(self.trainDataX)*numTrainRatio)
            print("after duplicating #training = %d, #positive = %d" % (numTrainSamples, self.trainDataY[:numTrainSamples].count(1)))
            #self.train_HS = int(self.trainDataY[:len(self.trainDataX)].count(1))
        
        self.trainDataX = np.asarray(self.trainDataX)
        self.trainDataY = np.asarray(self.trainDataY)
        #train_HS = self.trainDataY[:len(self.trainDataX)].count(1)
        
        print ('data preparation takes time(sec): %.4f' %(time()-tt))
        return (self.trainDataX, self.trainDataY)  
      
    """
    flip pattern left and right
    """
    def fliplr(self, image): 
        return np.fliplr(image)

    """
    flip pattern up and down
    """
    def flipud(self, image): 
        return np.flipud(image)         
    
    ''' unpack'''
    def unpack_data(self, data):
        dataX = np.unpackbits(data).reshape([-1,128*128])
        return dataX
	
def load_testdata(file2,numTestRatio = 1):
    testDataX = []
    testDataY = []
    print("reading %s" % (file2))
    with gzip.open(file2, "rb") as f:
                dataX, dataY = pickle.load(f)
                print("%d samples" % (len(dataX)))
                testDataX.extend(dataX)
                testDataY.extend(dataY)
    numTestSamples = int(len(testDataX)*numTestRatio)
    print(" #testing  = %d, #positive = %d" % (numTestSamples, testDataY[:numTestSamples].count(1)))
    testDataX = np.asarray(testDataX)
    testDataY = np.asarray(testDataY)
    #train_HS = self.trainDataY[:len(self.trainDataX)].count(1)
    return (testDataX, testDataY)  
def unpack_testdata(data):
        dataX = np.unpackbits(data).reshape([-1,128*128])
        return dataX
