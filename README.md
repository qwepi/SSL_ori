# SSL

## Dataset

Feature Tensor Extraction Data is already within this repo, original images can be found at http://appsrv.cse.cuhk.edu.hk/~hyyang/files/iccad-official.tgz

## Dependencies

numpy, tensorflow (tested on 1.3 and 1.9), pandas, json, ConfigParser, progress

## Train

e.g. to train iccad2 with 10% labeled samples, you need to modify iccad2\_config.ini

set ```save_path=./models/iccad2/ssl/```

set ```aug=0```

set ```train_ratio=0.1```

set ```aug=0```

and 

```python train_SSL_release.py iccad2_config.ini```

## Test

e.g. to test iccad2, you need to modify iccad1\_config.ini

set ```model_path=./models/iccad2/ssl/model.ckpt```

set ```aug=0``` and

```python train_SSL.py iccad2_config.ini```

## Batch Process

e.g. to train and test iccad2 with 10%, 30%, 50% labeled samples and different random seeds(50,100,150), you need to modify run.sh as folows:

for b in 2: do
for train_p in 0.1 0.3 0.5; do
for seed in 50 100 150; do

and 

```source run.sh```

then when all the runnings are done, go to folder "log_SSL" to check the testing results.


