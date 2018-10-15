##
# @file   gen_ini.py
# @author Yibo Lin
# @date   Jul 2018
#

import sys

initial_ini = """
[dir]
#benchmark path for training set
train_path = ./benchmarks/BENCHMARK/train
#benchmark path for one-fold cross validation
#val_path   = 
#benchmark path for model testing
test_path  = ./benchmarks/BENCHMARK/test

#path to save model
#save_path  = ./models/BENCHMARK/blaug/
#save_path  = ./models/BENCHMARK/bblaug/
#save_path  = ./models/BENCHMARK/bl/
#save_path  = ./models/BENCHMARK/bbl/
save_path = SAVE_PATH
#path to trained model

#model_path = ./models/BENCHMARK/blaug/model.ckpt
#model_path = ./models/BENCHMARK/bblaug/model.ckpt
#model_path = ./models/BENCHMARK/bl/model.ckpt
#model_path = ./models/BENCHMARK/bbl/model.ckpt
model_path = MODEL_PATH

[feature]
#the length of feature tensor
ft_length  = 32
block_dim  = 12
#whether do augmentation as described in Section IV-E
aug = 0 
data_flip = 0

forDAC = 0
train_ratio = TRAIN_RATIO
seed = SEED 
b = bB
"""

b = int(sys.argv[1])
train_p = float(sys.argv[2])
seed = int(sys.argv[3])
ini_filename = str(sys.argv[4])

ini = initial_ini.replace("BENCHMARK", "iccad%d" % (b))
ini = ini.replace("SAVE_PATH", "models/iccad%d/unlossfix_IPSL_m10000_p%g_seed%d/" % (b, train_p, seed))
#ini = ini.replace("SAVE_PATH", "models/iccad%d/losebased_altest_dacdecaybias_train_p%g_seed%d/" % (b, train_p, seed))
#ini = ini.replace("SAVE_PATH", "models/iccad%d/dac_train_p%g_seed%d/" % (b, train_p, seed))
if not train_p == 1.0:
    ini = ini.replace("MODEL_PATH", "models/iccad%d/formodel_SSL_dacdecaybias_p%g_seed%d/model-t3-p%g-s%d-step9999-SSL.ckpt" % (b, train_p, seed,train_p,seed))
else:
    ini = ini.replace("MODEL_PATH", "models/iccad%d/formodel_SSL_dacdecaybias_p%g_seed%d/model-t0-p%g-s%d-step9999-SSL.ckpt" % (b, train_p, seed,train_p,seed))
#ini = ini.replace("MODEL_PATH", "models/iccad%d/dac_train_p%g_seed%d/model-9999-0.3-.ckpt" % (b, train_p, seed))
ini = ini.replace("TRAIN_RATIO", "%g" % (train_p))
ini = ini.replace("SEED", "%d" % (seed))
ini = ini.replace("bB", "%d" % (b))

with open(ini_filename, "w") as f:
    f.write(ini)
