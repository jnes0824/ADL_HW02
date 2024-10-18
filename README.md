## Description 
This is homework 2 of applied deep learning 2024 fall.
See NTU ADL2024-HW2.pptx for details.

## Environment Set up
conda should be installed
1. create a conda environment with
1. `git clone https://github.com/jnes0824/ADL_HW02.git`
2. `cd ADL_HW02`
4. `conda activate ./env`(assume the env is created in the working dir)
5. install the correct version of pytorch(cuda 12.2) `conda install pytorch pytorch-cuda=12.1 -c pytorch-nightly -c nvidia`
6. `pip install -r requirements.txt`
```
git clone https://github.com/deankuo/ADL24-HW2.git
cd ADL24-HW2
pip install -e tw_rouge
```
10. check the dependency by running `python test.py`

## download data
1. `mkdir data`
1. `cd data`
1. `pip install gdown`
2. `gdown https://drive.google.com/uc?id=1t8QSuHXz7L9nYRrAwLQ4ponSweAp2WW_`
3. `unzip data.zip`

## Preprocessing Data
1. `python convert_to_json.py`

## training
1. download and login to wandb
```
pip install wandb
wandb login
```
2. start training
```
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix  "summarize: " \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --num_train_epochs 15\
    --checkpointing_steps epoch\
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_source_length 2048 \
    --output_dir ./model \
    --report_to wandb \
    --num_beams 4 \
    --with_tracking \
```

## download fine tuned model
`bash ./download.sh`

## inference
`python ./ADL24-HW2/eval.py -r ./data/public.jsonl -s ./data/submission.jsonl`
or
`bash ./run.sh`

## generation strategy
- 更改run_simulation.py中的strategies list，一次跑一項，多於一項會out of memory
`python run_simulation.py --model_path model`
- 結果在存在`rouge_results.jsonl`，以`python create_table.py ` 畫表格