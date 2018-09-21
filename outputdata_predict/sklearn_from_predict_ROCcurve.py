# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 23:24:11 2018

@author: Ying
"""

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
plt.rc('text', usetex=False) 
matplotlib.rcParams['text.latex.unicode']=False
plt.rc('font', family='serif')
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter
import numpy as np
import pdb

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x,axis=1)
    for i in range(len(exp_x)):
        exp_x[i]=exp_x[i]/sum_exp_x[i]
    #x_ones = np.ones_like(x)
    #sum_exp_x_division = sum_exp_x * x_ones
    #softmax_x = exp_x/sum_exp_x_division
    return exp_x


ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
#ratio = [0.1]
seed = [50, 100, 150, 200, 250]
#seed = [250]
bB = [2, 3, 4, 5]
#bB = [2]
#e_threshold = range(-90,95,10)
color=['r','--b','-.g']
#color_s = ['r','--b','-.g','--g','--r']

#class_forDAC = ["DAC"]
class_forDAC = ["DAC","SSL", "SPAL"]

#merge figures

#fig = plt.figure(num =24, figsize=(12,10),dpi=300)
fig = plt.figure(num =24,figsize=(12,11), dpi=300)
for b in range(len(bB)):
    txtname_y = "benchmark%d_testY.txt"%bB[b]
    y_output = np.loadtxt(txtname_y)
    for p in range(len(ratio)):
        #whole_TPR=[]
        #whole_FPR=[]
        ymajorFormatter = FormatStrFormatter('%1.1f')
        #fig = plt.figure(figsize=(3, 2.2))
        #ax = plt.gca()
        ax = fig.add_subplot(len(ratio),len(bB),b+4*p+1)
           
        for i in range(len(class_forDAC)):
            #p_TPR = np.zeros(11)
            #p_FPR = np.zeros(11)
            #TPR=np.zeros(len(e_threshold))
            #FPR=np.zeros(len(e_threshold))
            for s in seed:
                txtname = class_forDAC[i]+ "-benchmark%d-p%g-s%d.txt"%(bB[b], ratio[p], s)
                #txtname_FPR = class_forDAC[i] + "-FPR" + "-p%g-s%d-benchmark%d.txt"%(p, s, b)
                predict_output = np.loadtxt(txtname)
                #TPR_s = np.zeros(len(e_threshold))
                #FPR_s = np.zeros(len(e_threshold))
                #pdb.set_trace()
                predict_output_softmax = softmax(predict_output)
                y_pre = [y for (x,y) in predict_output_softmax]
                y_pre = np.array(y_pre)
                #pdb.set_trace()
                
                #plot for different seeds
                fpr, tpr,_ = roc_curve(y_output,y_pre)
                print("for",txtname)
                print("len(fpr)=",len(fpr))
                print("len(tpr)=",len(tpr))
            #plt.plot(fpr,tpr,color[i],label=class_forDAC[i])

'''
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        title = "b%d-p%g-seed250"%(bB[b],ratio[p])
        plt.title(title)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.xlim(0.013,0.018)
        #plt.ylim(0.85,1.00)
        #plt.yaxis.set_major_formatter(ymajorFormatter)
        #plt.legend(loc = 4)
        plt.xlim(-0.002,0.051)
        #plt.ylim(0.84,1.00)
        my_x_ticks = np.arange(0,0.051,0.01)
        #my_y_ticks = np.arange(0.84,1.02,0.04)    
        plt.xticks(my_x_ticks)
        #plt.yticks(my_y_ticks)
        plt.tight_layout()
        #file_save = "./figs/cROCcurve-seed50-b%d-p%g.pdf"%(bB[b],ratio[p])
        plt.show()
        #plt.savefig(file_save)
        print("b%d-p%g done"%(bB[b],ratio[p]))

file_save = "./figs/cROCcurve-seed250-x0.05-whole-b%d-p%g.pdf"%(bB[b],ratio[p])
#plt.savefig(file_save)


                if s == 50:
                    y_pre_whole = y_pre
                else:
                    y_pre_whole = y_pre_whole + y_pre
            y_pre_average = y_pre_whole/5
            #pdb.set_trace()

            fpr, tpr, threshold = roc_curve(y_output,y_pre_average)
            #print("fpr=",fpr)
            #print("tpr=",tpr)
            plt.plot(fpr,tpr, color[i], label=class_forDAC[i])

        #plt.plot(whole_FPR[0],whole_TPR[0],'r',label='DAC')
        #plt.plot(whole_FPR[1],whole_TPR[1], '--b', label='SSL')
        #plt.plot(whole_FPR[2],whole_TPR[2], '-.g', label='SPAL')
        #fill_mean(y_b2_mean,S_b2_mean,y_b2_std,S_b2_std,D2_m,D2_s)
        #fill_mean_two(S2_m,S2_s,D2_m,D2_s)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            title = "b%d-p%g"%(bB[b],ratio[p])
            plt.title(title)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            #plt.xlim(0.013,0.018)
            #plt.ylim(0.85,1.00)
            #plt.yaxis.set_major_formatter(ymajorFormatter)
            #plt.legend(loc = 4)
            plt.xlim(-0.002,0.051)
            #plt.ylim(0.84,1.00)
            my_x_ticks = np.arange(0,0.051,0.01)
            #my_y_ticks = np.arange(0.84,1.02,0.04)    
            plt.xticks(my_x_ticks)
            #plt.yticks(my_y_ticks)
            plt.tight_layout()
            file_save = "./figs/seed_cROCcurve-b%d-p%g.pdf"%(bB[b],ratio[p])
            plt.show()
            plt.savefig(file_save)
            print("b%d-p%g done"%(bB[b],ratio[p]))
#plt.savefig("./figs/bROC_whole.pdf")

        #legend
        fig_leg = plt.figure(figsize =(3, 0.8))
        #ax_leg = fig_leg.add_subplot(111)
        ax_leg = plt.gca()
        # add the legend from the previous axes
        ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3, frameon=False)
        # hide the axes frame and the x/y labels
        ax_leg.axis('off')
        plt.tight_layout()
        ax_leg.figure.savefig("./figs/legend_ROC.pdf")



                #for e in range(len(e_threshold)):
                    
                    #predict_threshold = [(x-e_threshold[e]*0.01,y) for (x,y) in predict_output]
                    #y_threshold = np.argmax(predict_threshold, axis=1)
                    tmp = y_threshold + y_output
                    chs = sum(tmp==2)
                    cnhs = sum(tmp==0)
                    ahs = sum(y_output)
                    anhs = sum(y_output==0)
                    TPR_s[e] = 1.0*chs/ahs
                    fs = anhs-cnhs
                    FPR_s[e] = 1.0*fs/anhs
                TPR = TPR + TPR_s
                FPR = FPR + FPR_s
            TPR = TPR/5
            FPR = FPR/5
            whole_TPR.append(TPR)
            whole_FPR.append(FPR)  
        #plot
        print("whole_TPR = ", whole_TPR)
        print("whole_FPR = ", whole_FPR)
        ymajorFormatter = FormatStrFormatter('%1.2f')
        fig = plt.figure(figsize=(3, 2.2))
        ax = plt.gca()
        plt.plot(whole_FPR[0],whole_TPR[0],'r',label='DAC')
        plt.plot(whole_FPR[1],whole_TPR[1], '--b', label='SSL')
        plt.plot(whole_FPR[2],whole_TPR[2], '-.g', label='SPAL')
        #fill_mean(y_b2_mean,S_b2_mean,y_b2_std,S_b2_std,D2_m,D2_s)
        #fill_mean_two(S2_m,S2_s,D2_m,D2_s)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        title = "b%d-p%g"%(b,p)
        plt.title(title)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #plt.xlim(0.013,0.018)
        #plt.ylim(0.85,1.00)
        #plt.yaxis.set_major_formatter(ymajorFormatter)
        #plt.legend(loc = 4)
        #plt.xlim(0.1,1.0)
        #plt.ylim(0.84,1.00)
        #my_x_ticks = np.arange(0.2,1.1,0.2)
        #my_y_ticks = np.arange(0.84,1.02,0.04)    
        #plt.xticks(my_x_ticks)
        #plt.yticks(my_y_ticks)
        plt.tight_layout()
        file_save = "figs/ROCcurve-b%d-p%g.pdf"%(b,p)
        plt.show()
        plt.savefig(file_save)
'''
