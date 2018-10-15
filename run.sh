##
# @file   run_baseline.sh
# @author Yibo Lin
# @date   Jul 2018
#
#!/bin/bash

mkdir -p log_unlossfix_dac_IPSL_test

for b in 4; do
#for b in 4 5; do
for train_p in 0.1; do 
#for train_p in 0.3 0.5 0.7 0.9; do 
#for seed in 50 100 150 200 250 300 350 400 450 500; do 
for seed in 50 100 150 200 250; do 
#for seed in 150; do
    ini="iccad${b}_config.ini"

    #if [[ ${seed} != 50 && ${train_p} != 0.5 ]]; then
        #continue
    #fi 

    python gen_ini_2.py ${b} ${train_p} ${seed} ${ini}

    log="log_unlossfix_dac_IPSL_test/unlossfix_IPSL_m10000_train_p${train_p}_seed${seed}.${ini}.log"

    echo train_p = ${train_p}, seed = ${seed}, log = ${log}
    cat ${ini} > ${log}
    python ./train_ISPL_al.py ${ini} >> ${log}
    #make iccad${b}_test >> ${log}

done 
done
done
