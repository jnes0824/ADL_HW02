python run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train_split.json \
    --validation_file ./data/validation.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --weight_decay 0.01 \
    --num_train_epochs 15 \
    --checkpointing_steps epoch\
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir ./output_mt5 \
    --report_to wandb \
    --num_beams 4 \
    --with_tracking \

python ./ADL24-HW2/eval.py -r ./data/public.jsonl -s ./data/submission.jsonl

python ./ADL24-HW2/eval.py -r ./data/public.jsonl -s ./data/summary_output3.jsonl 