#!/bin/bash

dataset_path="datasets/dataset2/"
datasets=(mico)  # Dataset list
# patterns=(P1 P2 P5 P6)  # Test pattern list
# P1 to P24
patterns=(P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19 P20 P21 P22 P23 P24)  # Test pattern list

# Output files
summary_file="log/G2Miner_summary_table.log" # Runtime result table
log_file="log/G2Miner_output.log" # Execution output

# Select GPU ID
export CUDA_VISIBLE_DEVICES=0

column_width=20
column_width_1=8
column_width_2=13


echo -e "Runtime of G2Miner and G2Miner + LUT (sec)" > $summary_file
echo -e "+------------------------------------------------------------------------------------+" >> $summary_file

header="| Pattern  | Optimization "
# for dataset in "${datasets[@]}"; do
for dataset in "$dataset_path"*; do
  dataset=$(basename "$dataset")
  header+="| $(printf "%-${column_width}s" $dataset)"
done
echo "$header" >> $summary_file
echo -e "+------------------------------------------------------------------------------------+" >> $summary_file

echo -e "Execution log started at $(date)" > $log_file

for pattern in "${patterns[@]}"; do
  row="| $(printf "%-${column_width_1}s" $pattern) | $(printf "%-${column_width_2}s" "None")"
  
  # for dataset in "${datasets[@]}"; do
  for dataset in "$dataset_path"*; do
    data_set=$(basename "$dataset")
    echo "data_set: $data_set"
    # command="./bin/pattern_gpu_GM ./datasets/$dataset/graph $pattern"
    command="./bin/pattern_gpu_GM $dataset/graph $pattern"
    
    output=$($command 2>&1)
    
    echo -e "\n====================\nRunning command: $command\n" >> $log_file
    echo "$output" >> $log_file
    
    runtime=$(echo "$output" | grep -oP "runtime \[G2Miner\] = \K\d+\.\d+")
    count=$(echo "$output" | grep -oP "Pattern \S+ count: \K\d+")
    
    if [[ -n "$runtime" && -n "$count" ]]; then
      row+="| $(printf "%-${column_width}.6f" $runtime)" 
    else
      row+="| $(printf "%-${column_width}s" "Error")"
    fi
  done

  echo "$row" >> $summary_file

  row_lut="| $(printf "%-${column_width_1}s" $pattern) | $(printf "%-${column_width_2}s" "LUT")"
  
  # for dataset_lut in "${datasets[@]}"; do
  for dataset_lut in "$dataset_path"*; do
    # command_lut="./bin/pattern_gpu_GM_LUT ./datasets/$dataset_lut/graph $pattern"
    command_lut="./bin/pattern_gpu_GM_LUT $dataset_lut/graph $pattern"
    
    output_lut=$($command_lut 2>&1)
    
    echo -e "\n====================\nRunning command: $command_lut\n" >> $log_file
    echo "$output_lut" >> $log_file
    
    runtime_lut=$(echo "$output_lut" | grep -oP "runtime \[G2Miner \+ LUT\] = \K\d+\.\d+")
    count_lut=$(echo "$output_lut" | grep -oP "Pattern \S+ count: \K\d+")
    
    if [[ -n "$runtime_lut" && -n "$count_lut" ]]; then
      row_lut+="| $(printf "%-${column_width}.6f" $runtime_lut)"
    else
      row_lut+="| $(printf "%-${column_width}s" "Error")"
    fi
  done

  echo "$row_lut" >> $summary_file

done

echo -e "+------------------------------------------------------------------------------------+" >> $summary_file

echo -e "\nExecution completed at $(date)" >> $log_file
