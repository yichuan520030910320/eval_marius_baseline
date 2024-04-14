datasets_name_list=("ogbn-papers100M" "mag240m" "friendster" "igb-full")
buffer_num_list=(1000 3000 5000)
models=("sage" "gat")

for dataset_name in "${datasets_name_list[@]}"; do
    echo "Processing dataset: ${dataset_name}"
    for model in "${models[@]}"; do
        echo "Processing model: ${model}"
        config_path="/nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gs.yaml"
        
        
        echo "Processing config: ${config_path}"
        
        for buffer_num in "${buffer_num_list[@]}"; do
            if [[ $model = "gat" && $buffer_num -eq 1000 ]]; then
                echo "Processing GAT model"
                config_path="/nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gat_flat.yaml"
            fi
            if [[ $model = "gat" && $buffer_num -ne 1000 ]]; then
                echo "Skipping due to model being GAT and buffer_num not equal to 1000"
                break  
            fi

            if [[ $model = "sage" && $dataset_name != "friendster"  ]]; then
                echo "Skipping due to model being sage but not fs"
                break  
            fi

            echo "Processing buffer_num: ${buffer_num}"
            sed -i "s/buffer_capacity: [0-9]*/buffer_capacity: $buffer_num/" "$config_path"
            marius_train "$config_path" > "/nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gs_mem_${buffer_num}_model_${model}_speed_128hidden.log"
            
            all_log="/nvme2n1/renjie/marius/change_mem_speed.log"
            echo "Processing dataset: ${dataset_name} buffer size : ${buffer_num} model : ${model}" >> "$all_log"
            python /nvme2n1/renjie/marius/cope_wspped.py "/nvme2n1/renjie/marius/datasets/my_${dataset_name}_8192/marius_gs_mem_${buffer_num}_model_${model}_speed.log" >> "$all_log" 
        done
    done
done
