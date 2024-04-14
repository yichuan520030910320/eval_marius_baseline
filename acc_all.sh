datasets_name_list=("ogbn-papers100M" "mag240m" "igb-full")
models=("sage" "gat")
## iterate over datasets
buffer_num_list=(1000 5000)
for dataset_name in ${datasets_name_list[@]}; do
    echo "Processing dataset: ${dataset_name}"
    ## iterate over model
    for model in ${models[@]}; do
        echo "Processing model: ${model}"
        config_path="/nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gs_acc.yaml"
        ## check if model is gat
        if [ $model == "gat" ]; then
            echo "Processing GAT model"
            config_path="/nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gs_gat_flat_acc.yaml"
        else
            echo "Processing SAGE model"
        fi
            
        echo "Processing config: ${config_path}"
        ## iterate over buffer_num
        for buffer_num in ${buffer_num_list[@]}; do
            echo "Processing buffer_num: ${buffer_num}"
            sed -i "s/buffer_capacity: [0-9]*/buffer_capacity: $buffer_num/" $config_path
            marius_train $config_path >> /nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gs_mem_${buffer_num}_model_${model}_acc_new.log
            ## deal with the log result
            all_log="/nvme2n1/renjie/marius/change_mem_acc.log"
            echo "Processing dataset: ${dataset_name} buffer size : ${buffer_num} model : ${model}" >> $all_log
            # python /nvme2n1/renjie/marius/cope_wspped.py /nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gs_mem_${buffer_num}.log >> $all_log 
        done
    done 
done