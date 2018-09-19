from model import *
import ConfigParser as cp
import sys
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[2])
from progress.bar import Bar

'''
Initialize Path and Global Params
'''
infile = cp.SafeConfigParser()
infile.read(sys.argv[1])

test_path   = infile.get('dir','test_path')


model_path = infile.get('dir','model_path')
fealen     = int(infile.get('feature','ft_length'))
blockdim   = int(infile.get('feature','block_dim'))
aug=int(infile.get('feature','AUG'))
seed = int(infile.get('feature','seed'))
train_ratio = float(infile.get('feature','train_ratio'))
bB = int(infile.get('feature','b'))
forDAC = int(infile.get('feature','forDAC'))   #for DAC or SSL


'''
Prepare the Input
'''
test_data = data(test_path, test_path+'/label.csv')
x_data = tf.placeholder(tf.float32, shape=[None, blockdim*blockdim, fealen])              #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label
x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                             #reshap to NHWC
x_ud   = tf.map_fn(lambda img: tf.image.flip_up_down(img), x)                   #up down flipped
x_lr   = tf.map_fn(lambda img: tf.image.flip_left_right(img), x)                #left right flipped
x_lu   = tf.map_fn(lambda img: tf.image.flip_up_down(img), x_lr)                #both flipped
predict_or = forward(x, is_training=False)                                      #do forward
predict_ud = forward(x_ud, is_training=False, reuse=True)  
predict_lr = forward(x_lr, is_training=False, reuse=True)
predict_lu = forward(x_lu, is_training=False, reuse=True)
if aug==1:
    predict = (predict_or + predict_lr + predict_lu + predict_ud)/4.0
else:
    predict = predict_or
#predict = predict_or
y      = tf.cast(tf.argmax(predict, 1), tf.int32)                                         
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                               #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
'''
Start testing
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver    = tf.train.Saver()
    saver.restore(sess, model_path)
    e_threshold = range(-10,11,2)
    chs = np.zeros(len(e_threshold))   #correctly predicted hs
    cnhs= np.zeros(len(e_threshold))   #correctly predicted nhs
    ahs = np.zeros(len(e_threshold))   #actual hs
    anhs= np.zeros(len(e_threshold))   #actual hs
    start   = time.time()
    bar = Bar('Detecting', max=test_data.maxlen/1000+1)
    for titr in xrange(0, test_data.maxlen/1000+1):
        if not titr == test_data.maxlen/1000:
            tbatch = test_data.nextbatch(1000, fealen)
        else:
            tbatch = test_data.nextbatch(test_data.maxlen-titr*1000, fealen)
        tdata = tbatch[0]
        tlabel= tbatch[1]
        tmp_label= np.argmax(tlabel, axis=1)
        tmp_predict    = predict.eval(feed_dict={x_data: tdata, y_gt: tlabel})
        #get ROC for different threshold
        #e_threshold = range(-0.1,0.1,0.02)
        #tmp_label= np.argmax(tlabel, axis=1)
        for i in range(len(e_threshold)):
            tmp_predict_threshold = [(x-e_threshold[i]*0.01,y) for (x,y) in tmp_predict]
            tmp_y_threshold = np.argmax(tmp_predict_threshold,axis=1)
            tmp      = tmp_label+tmp_y_threshold
            chs[i]  += sum(tmp==2)
            cnhs[i] += sum(tmp==0)
            ahs[i]  += sum(tmp_label)
            anhs[i] += sum(tmp_label==0)
        bar.next()
    bar.finish()
    print chs, ahs, cnhs, anhs
    hs_accu = np.zeros(len(e_threshold))
    fs  = np.zeros(len(e_threshold))
    #TPR = np.zeros(len(e_threshold))
    FPR = np.zeros(len(e_threshold))
    acc_ordinary = np.zeros(len(e_threshold))
    if forDAC ==1:
        txtname_TPR ="DAC-TPR-p%g-s%d-benchmark%d.txt"%(train_ratio, seed, bB)
        txtname_FPR = "DAC-FPR" + "-p%g-s%d-benchmark%d.txt"%(train_ratio, seed, bB)
    elif forDAC==0:
        txtname_TPR ="SSL-TPR-p%g-s%d-benchmark%d.txt"%(train_ratio, seed, bB)
        txtname_FPR = "SSL-FPR" + "-p%g-s%d-benchmark%d.txt"%(train_ratio, seed, bB)
    elif forDAC ==2:
        txtname_TPR ="AL-TPR-p%g-s%d-benchmark%d.txt"%(train_ratio, seed, bB)
        txtname_FPR = "AL-FPR" + "-p%g-s%d-benchmark%d.txt"%(train_ratio, seed, bB)
   
    #output, each line for each e_threshold(-0.1,0.1,0.02), whole output for one seed one ratio
    for i in range(len(e_threshold)):
            if not ahs[i] ==0:
                hs_accu[i] = 1.0*chs[i]/ahs[i]
            else:
                hs_accu[i] = 0
            fs[i]      = anhs[i] - cnhs[i]
            FPR[i] = 1.0*fs[i]/anhs[i]
            end       = time.time()
            acc_ordinary[i] = 1.0*(chs[i]+cnhs[i])/(ahs[i]+anhs[i])
            print("TPR=",hs_accu)
            print("FPR=",FPR)
            np.savetxt(txtname_TPR,hs_accu)
            np.savetxt(txtname_FPR,FPR)
            print("done saving for %d round" %i)   
