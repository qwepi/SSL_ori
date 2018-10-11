from model_ISPL import *
import ConfigParser as cp
import sys
import os
from datetime import datetime
import pickle
import gzip
import struct
import numpy as np
from array import array
import glob 
import pickle
import gzip
import time
import random
import pdb
from progress.bar import Bar
if len(sys.argv) > 2: 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])


'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(sys.argv[1])
train_path = infile.get('dir','train_path')
test_path   = infile.get('dir','test_path')
save_path = infile.get('dir','save_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
aug   = int(infile.get('feature','aug'))
train_ratio = float(infile.get('feature', 'train_ratio'))
seed = int(infile.get('feature', 'seed'))
bB = int(infile.get('feature','b'))

if seed != 0: 
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    print("set random seed to %d" % (seed))

#test
test_data = data(test_path, test_path+'/label.csv')

'''
Prepare the Optimizer
'''
if not train_ratio ==1:
    trainX,trainY,unX,unY = DCT_data(train_path, train_path+'/label.csv', preload=True, ratio=train_ratio)
    unX = np.reshape(unX,[-1,12,12,32])
else:
    trainX,trainY = DCT_data(train_path, train_path+'/label.csv', preload=True, ratio=train_ratio)

trainX = np.reshape(trainX,[-1,12,12,32])

train_original = list(zip(trainX,trainY))
x_data = tf.placeholder(tf.float32, shape=[None, blockdim,blockdim, fealen], name="x_data")              #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2], name="y_gt")                                      #ground truth label
P      = tf.placeholder(tf.float32,shape=[None,None], name="P")#pairwise constraint
W      = tf.placeholder(tf.float32,shape=[None,], name="W")#weight
fortest= tf.placeholder(tf.int32, name="fortest")# whether the forward_crosstask process is training or testing, affect dropout layer

x = x_data

if aug==1:
    predict, predict1 = forward_crosstask(x, flip=True)   
elif fortest == 0:
    predict,predict1 = forward_crosstask(x)
else:
    predict,predict1 = forward_crosstask(x, is_training = False)

#do forward
loss_w   = tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=predict) 
loss   = tf.reduce_mean(loss_w*W)                                                             #calc batch loss of classification stream
D_kl,C_loss = get_Dkl(predict1,P)                        #calc batch loss of clustering stream
y      = tf.cast(tf.argmax(predict, 1), tf.int32)                                         
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                                                    #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
gs     = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)       #define global step
lr_holder = tf.placeholder(tf.float32, shape=[])
optimizer    = tf.train.AdamOptimizer(lr_holder, beta1=0.9)
opt_cnn_clust    = optimizer.minimize(loss+C_loss, gs)
opt_cnn    = optimizer.minimize(loss, gs)
lr     = 0.001 #initial learning rate and lr decay
dr     = 0.65 #learning rate decay rate
maxitr = 301 # training steps for MTNN
maxitr_un = 301 # training steps for selecting unlabeled samples
bs     = 32   #training batch size
c_step = 1000 #display step
b_step = 3200 #lr decay step
t_step = 10000 #total step
num_S = 15 #unlabeled data subset number
t_num = 4 #iteration round
al_ratio = 0.01 #ratio parameter for ISPL data selecting
v_num = 4 # use whether un_loss(3) or v_loss(4) to sort unlabeled data subset
un_loss_based = 0 # use whether un_loss(1) or v_loss(0) to sort unlabeled data subset
ifISPL = 0 # use ISPL(1) or not(0)
ckpt   = True#set true to save trained models.

l_step =100
'''
Start the training
'''
t1 = time.time()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.44
global_variables_initializer = tf.global_variables_initializer()
with tf.Session(config=config) as sess:
    saver   = tf.train.Saver(max_to_keep=50000)
    weight_train = np.ones(len(trainX))
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    def run_train(trainX,trainY):
        chs = 0
        cnhs= 0   #correctly predicted nhs
        ahs = 0   #actual hs
        anhs= 0   #actual hs
        start   = time.time()
        num = 0
        for titr in xrange(0, len(trainX)/bs+1):
            if not titr == len(trainX)/bs:
                #tbatch = test_data.nextbatch(1000, fealen)
                tbatch,num = batchfor_test(trainX,trainY,num,bs)
            else:
                if not len(trainX)-titr*bs ==0:
                #tbatch = test_data.nextbatch(test_data.testlen-titr*1000, fealen)
                    tbatch,num = batchfor_test(trainX,trainY,num,len(trainX)-titr*bs)
                else:
                    break
            tdata = tbatch[0]
            tlabel= tbatch[1]
            tmp_y = y.eval(feed_dict={x_data: tdata, y_gt:tlabel,  fortest:1})
            tmp_label= np.argmax(tlabel, axis=1)
            tmp      = tmp_label+tmp_y
            chs += sum(tmp==2)
            cnhs+= sum(tmp==0)
            ahs += sum(tmp_label)
            anhs+= sum(tmp_label==0)
        print('For training data:')
        print chs, ahs, cnhs, anhs
        if not ahs ==0:
            hs_accu = 1.0*chs/ahs
        else:
            hs_accu = 0
        fs      = anhs - cnhs
        end       = time.time()
        acc_ordinary = 1.0*(chs+cnhs)/(ahs+anhs)
        print ahs, anhs
        print('(training)Hotspot Detection Accuracy is %f'%hs_accu)
        print('False Alarm is %f'%fs)
        print('Ordinary Accuracy is %f' %acc_ordinary)
        print('Test Runtime is %f seconds'%(end-start))
 
    def run_train_clu(trainX,trainY):
        chs = 0
        cnhs= 0   #correctly predicted nhs
        ahs = 0   #actual hs
        anhs= 0   #actual hs
        start   = time.time()
        num = 0
        for titr in xrange(0, len(trainX)/bs+1):
            if not titr == len(trainX)/bs:
                #tbatch = test_data.nextbatch(1000, fealen)
                tbatch,num = batchfor_test(trainX,trainY,num,bs)
            else:
                if not len(trainX)-titr*bs ==0:
                #tbatch = test_data.nextbatch(test_data.testlen-titr*1000, fealen)
                    tbatch,num = batchfor_test(trainX,trainY,num,len(trainX)-titr*bs)
                else:
                    break
            tdata = tbatch[0]
            tlabel= tbatch[1]
            #tmp_y = y.eval(feed_dict={x_data: tdata, y_gt:tlabel,  fortest:1})
            #Ying
            tmp_y,unloss_batch, predict1_train, Dkl_train = sess.run([y,loss_w,predict1,D_kl],feed_dict={x_data: tdata, y_gt:tlabel,  fortest:1})
            pdb.set_trace()
            tmp_label= np.argmax(tlabel, axis=1)
            pc_train_batch = pairwise_constraint(tdata,tmp_label)
            Dkl_np_train_batch, closs_train_batch = get_Dkl_C(predict1_train,pc_train_batch)
            wi_np_train_batch = get_wi(Dkl_np_train_batch,pc_train_batch)
            vloss_batch = wi_np_train_batch*unloss_batch
            pdb.set_trace()
            #tmp_label= np.argmax(tlabel, axis=1)
            tmp      = tmp_label+tmp_y
            chs += sum(tmp==2)
            cnhs+= sum(tmp==0)
            ahs += sum(tmp_label)
            anhs+= sum(tmp_label==0)
        print('For training data:')
        print chs, ahs, cnhs, anhs
        if not ahs ==0:
            hs_accu = 1.0*chs/ahs
        else:
            hs_accu = 0
        fs      = anhs - cnhs
        end       = time.time()
        acc_ordinary = 1.0*(chs+cnhs)/(ahs+anhs)
        print ahs, anhs
        print('(training)Hotspot Detection Accuracy is %f'%hs_accu)
        print('False Alarm is %f'%fs)
        print('Ordinary Accuracy is %f' %acc_ordinary)
        print('Test Runtime is %f seconds'%(end-start))
 

    def run_test():
        chs = 0   #correctly predicted hs
        cnhs= 0   #correctly predicted nhs
        ahs = 0   #actual hs
        anhs= 0   #actual hs
        start   = time.time()
        bar = Bar('Detecting', max=test_data.maxlen/1000+1)
        for titr in xrange(0, test_data.maxlen/1000+1):
            if not titr == test_data.maxlen/1000:
                tbatch = test_data.nextbatch(1000, fealen)
            else:
                tbatch = test_data.nextbatch(test_data.maxlen-titr*1000, fealen)
            tdata = tbatch[0]
            tlabel= tbatch[1]
            tdata = np.reshape(tdata,[-1,12,12,32])
            tmp_y    = y.eval(feed_dict={x_data: tdata, y_gt: tlabel})
            tmp_label= np.argmax(tlabel, axis=1)
            tmp      = tmp_label+tmp_y
            chs += sum(tmp==2)
            cnhs+= sum(tmp==0)
            ahs += sum(tmp_label)
            anhs+= sum(tmp_label==0)
            bar.next()
        bar.finish()
        print ('For test data:')
        print chs, ahs, cnhs, anhs
        if not ahs ==0:
            hs_accu = 1.0*chs/ahs
        else:
            hs_accu = 0
        fs      = anhs - cnhs
        if not fs+chs == 0:
            precision = 1.0*chs/(chs+fs)
        else:
            precision = 0
        end       = time.time()
        acc_ordinary = 1.0*(chs+cnhs)/(ahs+anhs)
        #if not hs_accu == 0 & precision ==0:
        F_score = 1.0*2*(hs_accu*precision)/(hs_accu+precision)
        #else:
            #F_score = 0
        print ahs, anhs, hs_accu, fs, precision, F_score
        print('Hotspot Detection Accuracy is %f'%hs_accu)
        print('False Alarm is %f'%fs)
        print('Ordinary Accuracy is %f' %acc_ordinary)
        print('F_score is %f' %F_score)
        print('Test Runtime is %f seconds'%(end-start))
   

    for t in xrange(t_num):
        print('%d round start' % (t+1))
        sess.run(global_variables_initializer)
        lr = 0.001
        if t ==t_num-1:
            maxitr = t_step
        if train_ratio ==1:
            maxitr = t_step
        print("*******cross task training*******")
        for step in xrange(maxitr):
            batch =  get_batch_withweight_bias(trainX1,trainY1,weight_train,bs)
            batch_data = batch[0]
            batch_label= batch[1]
            batch_wi   = batch[2]
            batch_nhs  = batch[3]
            batch_nhs_wi = batch[5]
            batch_label_all_without_bias = processlabel(batch_label)
            batch_label_nhs_without_bias = processlabel(batch[4])
            nhs_loss = loss.eval(feed_dict={x_data: batch_nhs, y_gt: batch_label_nhs_without_bias, W:batch_nhs_wi, fortest:0})
            if maxitr == t_step:
                if step < b_step:
                    delta1 = 0
                elif step < b_step*2:
                    delta1 = 0.15
                else:
                    delta1 = 0.30
            else:  
                if t ==0 or t == t_num-1:
                    delta1 = loss_to_bias(nhs_loss, 6, threshold=0.3)
                else:
                    delta1 = loss_to_bias(nhs_loss, 6, threshold=0.05)
            batch_label_all_with_bias = processlabel(batch_label, delta1 = delta1)
            pc_labeled = pairwise_constraint(batch_data,batch_label)
            feed_dict = {x_data: batch_data, y_gt: batch_label_all_with_bias, W:batch_wi, P: pc_labeled, lr_holder:lr, fortest:0}
            training_loss, training_acc = sess.run([loss, accu], feed_dict=feed_dict)
            feed_dict = {x_data: batch_data, y_gt: batch_label_all_with_bias, W:batch_wi, P: pc_labeled, lr_holder:lr, fortest:0}
            opt_cnn_clust.run(feed_dict = feed_dict)
            learning_rate = lr
            if not t ==t_num-1:
                if step % l_step == 0 and step >0:
                    format_str = ('%s: p %f, step %d, loss = %.2f, learning_rate = %f, training_accu = %f')
                    print (format_str % (datetime.now(), train_ratio, step, training_loss, learning_rate, training_acc))

            if t ==t_num-1:
                if step % c_step == 0 and step >0:
                    format_str = ('%s: p %f, step %d, loss = %.2f, learning_rate = %f, training_accu = %f')
                    print (format_str % (datetime.now(), train_ratio, step, training_loss, learning_rate, training_acc))
            if train_ratio == 1:
                if step % c_step == 0 and step >0:
                    format_str = ('%s: p %f, step %d, loss = %.2f, learning_rate = %f, training_accu = %f')
                    print (format_str % (datetime.now(), train_ratio, step, training_loss, learning_rate, training_acc))
            if step % b_step == 0 and step >0:
                lr = lr * dr
        
        run_train_clu(trainX,trainY)
        run_test()
        if t ==t_num-1:
            path = "%smodel-p%g-s%d-step%d.ckpt" % (save_path, train_ratio, seed, step)
            print("save to path:", path)
            saver.save(sess, path)
            print("all three iteration are done" )
            #run_test()
            break
      
        if train_ratio ==1:
            path = "%smodel-p%g-s%d-step%d.ckpt" % (save_path, train_ratio, seed, step)
            print("save to path:", path)
            saver.save(sess, path)
            print("all labeled data are used, no need to do the other part")
            break
    
        print("*******get pseudo labels and weight*******")
        #acc
        chs = 0   #correctly predicted hs
        cnhs= 0   #correctly predicted nhs
        ahs = 0   #actual hs
        anhs= 0   #actual hs
       
        un_p_label = []
        un_loss1 =[]
        un_yDk = []
        num = 0
        bar = Bar('getting pseduo label and weight', max=len(unX)/bs+1)
        for titr in xrange(0,len(unX)/bs+1):
            if not titr == len(unX)/bs:
                tbatch,num = batchfor_test(unX,unY,num,bs)
            else:
                if not len(unX)-titr*bs ==0:
                    tbatch,num = batchfor_test(unX,unY,num,len(unX)-titr*bs)
                else:
                    break
            tdata = tbatch[0]
            tlabel= tbatch[1]
            #pdb.set_trace()
            #tmp_y, y_Dk, loss_un = \
            #        y.eval(feed_dict={x_data: tdata,y_gt:tlabel, fortest:1}), \
            #        predict1.eval(feed_dict={x_data: tdata,y_gt:tlabel, fortest:1}),\
            #        loss_w.eval(feed_dict={x_data: tdata,y_gt:tlabel, fortest:1})
           
            tmp_y,y_predict,y_Dk = sess.run([y,predict,predict1],feed_dict={x_data: tdata, fortest:1})
            un_p_label = data_flatten(un_p_label,tmp_y)
            tmp_y_label = [vectorized_result(y_sample) for y_sample in tmp_y]
            #get Dkl and wi
            #pdb.set_trace()
            loss_un = softmax_cross_loss(tmp_y_label,y_predict)
            un_loss1 = data_flatten(un_loss1,loss_un)
            un_yDk = un_yDk + y_Dk.tolist()
            
            #get acc
            tmp_label= np.argmax(tlabel, axis=1)
            tmp      = tmp_label+tmp_y
            chs += sum(tmp==2)
            cnhs+= sum(tmp==0)
            ahs += sum(tmp_label)
            anhs+= sum(tmp_label==0)
            bar.next()
        bar.finish()
        #get Dkl and wi for whole unlabeled data
        #pc_un = pairwise_constraint(unX,un_p_label)
        #Dkl_un,Closs_un = get_Dkl_C(un_yDk,pc_un)
        #wi,di = get_wi(Dkl_un, pc_un)
        print chs, ahs, cnhs, anhs
        if not ahs ==0:
            hs_accu = 1.0*chs/ahs
        else:
            hs_accu = 0
        fs      = anhs - cnhs
        #loss_v = wi * un_loss1
        print ahs, anhs
        print('Unlabeled Data Prediction Accuracy is %f'%hs_accu)
        print('False Alarm is %f'%fs)
        
        #bar.next()
        #bar.finish()
        #get Dkl and wi for whole unlabeled data
        pc_un = pairwise_constraint(unX,un_p_label)
        Dkl_un,Closs_un = get_Dkl_C(un_yDk,pc_un)
        wi = get_wi(Dkl_un, pc_un)
        loss_v = wi*un_loss1
      
        print("*******get unlabeled data for next training, %d round*******" % num_S)
        #define r and choose data for next iteration
        #training model(need X, predict Y and weight)
        #Ying
        lr = 0.001
        un_v_loss = wi*un_loss1
        un_wholeXYwl = list(zip(unX,un_p_label,wi,un_loss1,un_v_loss))
        un_wholetest = list(zip(un_p_label,unY,wi,un_loss1,un_v_loss))
        #print("un_wholetest=",un_wholetest)
        txtname = './unlossfix_test/unlossfix-testDkl-trainclu-benchmard%d-p%g-s%d-t%d.txt'%(bB,train_ratio,seed,t)
        pdb.set_trace()
        #np.savetxt(txtname,un_wholetest)
        print("un_wholetest save to ", txtname)
        un_wholeXYwl.sort(key = lambda x:x[v_num])
        num_k = int(len(un_wholeXYwl)/num_S+1)
        un_batches = [un_wholeXYwl[k:k+num_k] for k in range(0, len(un_wholeXYwl), num_k)]
        un_for_r = []
        un_for_r_acc = []
        bar = Bar('training model with unlabeled data to define r', max=num_S)
        for i in range(num_S):
            print("use unlabeled data to train CNC:", i)
            un_for_r = un_for_r + un_batches[i]
            unX_r,un_p_r, un_weight_r = get_data_un_r(un_for_r)
            unX_r = np.array(unX_r)
            un_p_r = np.array(un_p_r)
            un_weight_r = np.array(un_weight_r)
            sess.run(global_variables_initializer)
            for step in xrange(maxitr_un):
                batch = get_batch_withweight(unX_r,un_p_r,un_weight_r,bs)
                batch_data = batch[0]
                batch_label= batch[1]
                batch_wi   = batch[2]
                batch_label_all_without_bias = processlabel(batch_label)
                pc_labeled = pairwise_constraint(batch_data,batch_label)
                feed_dict = {x_data: batch_data, y_gt: batch_label_all_without_bias, W:batch_wi, P:pc_labeled, lr_holder:lr, fortest:0}
                training_loss, training_acc, _ = sess.run([loss, accu, opt_cnn_clust], feed_dict=feed_dict)
                learning_rate = lr
                if step % b_step == 0 and step >0:
                    lr = lr * dr
            #get labeled acc based on unlabeled data set,use the one with minimum loss
            chs = 0   #correctly predicted hs
            cnhs= 0   #correctly predicted nhs
            ahs = 0   #actual hs
            anhs= 0   #actual hs
            start   = time.time()
            num = 0
            for titr in xrange(0, len(trainX)/bs+1):
                if not titr == len(trainX)/bs:
                    tbatch,num = batchfor_test(trainX,trainY,num,bs)
                else:
                    if not len(trainX)-titr*bs ==0:
                        tbatch,num = batchfor_test(trainX,trainY,num,len(trainX)-titr*bs)
                    else:
                        break
                tdata = tbatch[0]
                tlabel= tbatch[1]
                tmp_y = y.eval(feed_dict={x_data: tdata, y_gt:tlabel,  fortest:1})
                tmp_label= np.argmax(tlabel, axis=1)
                tmp      = tmp_label+tmp_y
                chs += sum(tmp==2)
                cnhs+= sum(tmp==0)
                ahs += sum(tmp_label)
                anhs+= sum(tmp_label==0)
            print chs, ahs, cnhs, anhs
            if not ahs ==0:
                hs_accu = 1.0*chs/ahs
            else:
                hs_accu = 0
            acc_whole = 1.0*(chs+cnhs)/(ahs+anhs)
            un_for_r_acc.append(acc_whole)
            bar.next()
        bar.finish()
        print("for %f labeled data" % (len(trainX)),", un_for_r =", un_for_r_acc)
        r_max = 0
        r_index = 0
        for i in range(len(un_for_r_acc)):
            if not un_for_r_acc[i] < r_max:
                r_max = un_for_r_acc[i]
                r_index = i
        print("r_max =%f, r_index =%d" % (r_max,r_index))
        r = un_batches[r_index][-1][v_num]
        r_al_num = int(len(unX)*al_ratio/(2**t))+1
        print("r_al_num = ", r_al_num)
        if not r_index ==14:
            r_al = un_batches[-1][-r_al_num][v_num]
            print("r_al = %f" %r_al)
        loss_v_zero_number = (loss_v.tolist()).count(0)
        if not loss_v_zero_number == 0:
            print("loos_v has %d zeros" % (loss_v_zero_number))
        else:
            print("no zero in loss_v")
        print("r=",r)
        print('\n')
        vi_un = np.zeros_like(loss_v)

        if not un_loss_based == 1:
            for i in range(len(loss_v)):
                 if loss_v[i] <= r:
                    vi_un[i] = 1
                #ISPL
                 if ifISPL == 1:
                     if not r_index ==14:
                         if loss_v[i] >= r_al:
                             vi_un[i] = -1
        else:
            for i in range(len(un_loss1)):
                if un_loss1[i] <= r:
                    vi_un[i] =1
                if ISPL ==1:
                    if not r_index ==14:
                        if un_loss1[i] >= r_al:
                            vi_un[i] = -1
            
        unlabel_whole = list(zip(unX,un_p_label))
        unlabel_fortrain = []
        w_un_fortrain = []
        unlabel_al = []
        w_un_al =[]
        for i in range(len(vi_un)):
            if vi_un[i] == 1:
                unlabel_fortrain.append(unlabel_whole[i])
                w_un_fortrain.append(wi[i])
            #ISPL, inverse labels for low confident samples
            if ifISPL ==1:
                if vi_un[i] == -1:
                    unlabel_al.append(unlabel_whole[i])
                    w_un_al.append(wi[i])
        #ISPL, label and weight assignment
        if ifISPL ==1:
            if not r_index ==14:
                print("unlabel_al number:", len(unlabel_al))
                unlabel_al = data_flatten2(unlabel_al)
                X_al,Y_al = data_split_sencond(unlabel_al)
                Y_al_ones = np.ones_like(Y_al)
                Y_al_new = Y_al_ones - Y_al
                w_al_ones = np.ones_like(w_un_al)
                w_al_new = w_al_ones
                unlabel_al_new = list(zip(X_al,Y_al_new))
                newdata_fortrain = train_original + unlabel_fortrain+unlabel_al_new
                print("%d unlabeled data for ISPL are used" % (len(unlabel_al_new)))
            else:
                newdata_fortrain = train_original + unlabel_fortrain
            print("%d unlabeled data for spl are used" % (len(unlabel_fortrain)))
            newdata_fortrain = data_flatten2(newdata_fortrain)
            newdataX, newdataY = data_split_sencond(newdata_fortrain)
            len_trainingX = len(trainX)
            if not r_index ==14:
                #w_original
                w_new = np.ones(len_trainingX).tolist() + w_un_fortrain + w_un_al
            else:
                w_new = np.ones(len_trainingX).tolist() + w_un_fortrain
        else:
            newdata_fortrain = train_original + unlabel_fortrain
            newdata_fortrain = data_flatten2(newdata_fortrain)
            newdataX, newdataY = data_split_sencond(newdata_fortrain)
            len_trainingX = len(trainX)
            print("%d unlabeled data for spl are used" % (len(unlabel_fortrain)))
            w_new = np.ones(len_trainingX).tolist() + w_un_fortrain
        print("%d data totally are used" % (len(newdataX)))
        trainX1 = np.array(newdataX)
        trainY1 = np.array(newdataY)
        weight_train = np.array(data_flatten2(w_new))
        print("round ok at",t+1)
        print('\n')
print("training time is(seconds):", time.time()-t1)
    
    
