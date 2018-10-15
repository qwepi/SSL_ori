# DLHSD

## Dataset

Feature Tensor Extraction Data is already within this repo, original images can be found at http://appsrv.cse.cuhk.edu.hk/~hyyang/files/iccad-official.tgz

## Dependencies

numpy, tensorflow (tested on 1.3 and 1.9), pandas, json, ConfigParser, progress

## Train

e.g. to train iccad2 with 10% labeled samples, you need to modify iccad2\_config.ini

set ```save_path=./models/iccad2/ssl/```

set ```aug=0```

set ```train_ratio =0.1```

set ```aug=0```

and 

```python train_SSL_release.py iccad2_config.ini```

## Test

e.g. to test iccad2, you need to modify iccad1\_config.ini

set ```model_path=./models/iccad2/ssl/model.ckpt```

set ```aug=0``` and

```python train_SSL.py iccad2_config.ini```


