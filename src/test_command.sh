##########################################################################################
#                      generate pseudo stance labels by BART-MNLI
##########################################################################################
# bash ./test_command.sh ../config/config-bert.txt > train_TTS_openworld_result.txt

export PYTHONHASHSEED=1
for val in "vast_data"
do
    echo "Now it's on target ${val}"
    train_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv
    dev_data=../data_all/raw_val_all_onecol.csv
    test_data=../data_all/raw_test_all_onecol.csv
    kg_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv
    pseudo_data=../data_all/raw_train_all_generated_kg_zerotrn.csv
    config_data=../config/config-gen.txt

    rm ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
    cp ../data_all/raw_train_all_subset_kg_epoch_led_onecol.csv ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
    for epoch in {0..0}
    do
        echo "start training Gen ${epoch}......"
        python generation.py -c ${config_data} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} \
                              -g ${epoch} -s 1 -d 0.7 --exclude ${val} -pseudo ${pseudo_data} -t vast
    done
    
    train_data=../data_all/raw_train_all_generated_kg_zerotrn.csv
    dev_data=../data_all/raw_val_all_onecol.csv
    test_data=../data_all/raw_test_all_onecol.csv
    kg_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv

    rm ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
    cp ../data_all/raw_train_all_subset_kg_epoch_led_onecol.csv ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
    for seed in {0..3}
    do
        for epoch in {0..1}
        do
            echo "start training Gen ${epoch}......"
            python train_model.py -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} --exclude ${val} \
                                  -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3 --earlystopping_step 5 -p 100 -z -t vast -n 1
        done
    done
done

# export PYTHONHASHSEED=1
# for val in "mask" "fauci" "home" "school"
# do
#     echo "Now it's on target ${val}"
#     train_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     dev_data=../data_all/raw_val_all_onecol.csv
#     test_data=../data_all/raw_test_all_onecol.csv
#     kg_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     pseudo_data=../data_all/raw_train_all_generated_kg_zerotrn.csv
#     config_data=../config/config-gen.txt

#     rm ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     cp ../data_all/raw_train_all_subset_kg_epoch_led_onecol.csv ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     for epoch in {0..0}
#     do
#         echo "start training Gen ${epoch}......"
#         python generation.py -c ${config_data} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} \
#                               -g ${epoch} -s 1 -d 0.7 --exclude ${val} -pseudo ${pseudo_data} -t covid
#     done
    
#     train_data=../data_all/raw_train_all_generated_kg_zerotrn.csv
#     dev_data=../data_all/raw_val_all_onecol.csv
#     test_data=../data_all/raw_test_all_onecol.csv
#     kg_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv

#     rm ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     cp ../data_all/raw_train_all_subset_kg_epoch_led_onecol.csv ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     for seed in {0..3}
#     do
#         for epoch in {0..1}
#         do
#             echo "start training Gen ${epoch}......"
#             python train_model.py -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} --exclude ${val} \
#                                   -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3 --earlystopping_step 5 -p 100 -z -t covid -n 2
#         done
#     done
# done

# export PYTHONHASHSEED=1
# for val in "atheism" "climate" "feminist" "hillary" "abortion"
# do
#     echo "Now it's on target ${val}"
#     train_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     dev_data=../data_all/raw_val_all_onecol.csv
#     test_data=../data_all/raw_test_all_onecol.csv
#     kg_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     pseudo_data=../data_all/raw_train_all_generated_kg_zerotrn.csv
#     config_data=../config/config-gen.txt

#     rm ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     cp ../data_all/raw_train_all_subset_kg_epoch_led_onecol.csv ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     for epoch in {0..0}
#     do
#         echo "start training Gen ${epoch}......"
#         python generation.py -c ${config_data} -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} \
#                               -g ${epoch} -s 1 -d 0.7 --exclude ${val} -pseudo ${pseudo_data} -t semeval
#     done
    
#     train_data=../data_all/raw_train_all_generated_kg_zerotrn.csv
#     dev_data=../data_all/raw_val_all_onecol.csv
#     test_data=../data_all/raw_test_all_onecol.csv
#     kg_data=../data_all/raw_train_all_subset_kg_epoch_onecol.csv

#     rm ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     cp ../data_all/raw_train_all_subset_kg_epoch_led_onecol.csv ../data_all/raw_train_all_subset_kg_epoch_onecol.csv
#     for seed in {0..3}
#     do
#         for epoch in {0..1}
#         do
#             echo "start training Gen ${epoch}......"
#             python train_model.py -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} --exclude ${val} \
#                                   -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3 --earlystopping_step 5 -p 100 -z -t semeval -n 4
#         done
#     done
# done