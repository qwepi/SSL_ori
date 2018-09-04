##
# @file   run_baseline.sh
# @author Yibo Lin
# @date   Jul 2018
#
#!/bin/bash

mkdir -p log_dac

#for b in 2 3 4 5; do
for b in 2 ; do
for train_p in 0.1 0.3 0.5 0.7 0.9 1.0; do 
#for train_p in 0.1 0.3 0.5; do 
#for seed in 50 100 150 200 250 300 350 400 450 500; do 
for seed in 50 100 150 200 250; do 
    ini="iccad${b}_config.ini"

    #if [[ ${b} == 4 && ${train_p} != 0.1 ]]; then
        #continue
    #fi 

    python gen_ini.py ${b} ${train_p} ${seed} ${ini}

    log="log_dac/DAC_thres005_train_p${train_p}_seed${seed}.${ini}.log"

    echo train_p = ${train_p}, seed = ${seed}, log = ${log}
    cat ${ini} > ${log}
    python ./train_dac.py ${ini} >> ${log}
    make iccad${b}_test >> ${log}
	#python ./test.py ${ini} >> ${log}

done 
done
done
