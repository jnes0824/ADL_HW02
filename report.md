---
title: ADL HW02

---

## Model (1%)
### Describe the model architecture and how it works on text summarization.
- mT5 的架構類似於T5，兩者都基於編碼器-解碼器（Encoder-Decoder）的 Transformer 架構，並且都採用了“text-to-text”的統一任務格式，但mt5使用了擴展版的 mC4 數據集，這個數據集覆蓋了 101 種語言。
-  在預訓練階段，mT5 使用Masked Language Modeling方法進行訓練，即隨機選取輸入文本中的一些片段進行掩蔽，然後讓模型預測被掩蔽的詞。
-  編碼階段（Encoding Stage）： 輸入文本首先會經過編碼器，編碼器將輸入的序列（如要進行摘要的文章）轉換為一系列隱藏向量（hidden states）。
-  解碼階段（Decoding Stage）： 解碼器則從編碼器產生的隱藏向量中生成目標序列（如文章的摘要）。解碼過程是逐步進行的，即解碼器會生成一個詞，然後基於該詞和隱藏向量再生成下一個詞，直到生成完整的摘要。
-   mT5 的 "text-to-text" 框架，在pretraining時即對不同任務用不同prefix來做訓練，如'summarize'，因此fine tuning時使用prefix 'summarize: '來指定任務為生成摘要，能夠幫助模型理解接下來要進行的任務
-   https://arxiv.org/pdf/1910.10683 中有提到，prefix也是一種hyperparameter，但他們發現在這個模型上影響不大
## Preprocessing 
### Describe your preprocessing (e.g. tokenization, data cleaning and etc.)
- 使用SentencePiece分詞器:設計目的是解決傳統分詞方法在多語言或無空格語言（如中文、日文）中的局限性，並簡化 NLP 系統中的預處理和後處理過程。SentencePiece 是語言獨立的，通過將輸入文本視為純字符序列進行分詞，完全無需依賴詞語邊界或空格，這使其能夠應對各種語言。
- mT5 使用 SentencePiece 的 Unigram Language Model 進行子詞分割。Unigram 模型從一個預定義的詞彙表開始，逐步移除不常見的子詞，保留那些能夠更好地表示語言的子詞單元。
- 為了處理未見過的罕見字符或低資源語言，mT5 tokenizer 使用了字節回退機制。當分詞器遇到無法識別的字符時，它會將這些字符以單個字節的形式進行處理，保證輸入的每個字符都能被有效分割並映射到詞彙表中。
- Truncation and Padding：對title和maintext進行Truncation，max_length設為128和1024。長度超過指定最大值的輸入會被截斷，長度不足的會進行填充。在進行標記化時，truncation=True 會確保超出最大長度的文本被截斷，而 padding 參數會確保文本對齊，短的句子會填充到與其他輸入相同的長度。
- ignore_pad_token_for_loss: 控制模型在計算損失時是否忽略填充 token，填充的部分會被忽略，以避免它們對損失計算產生不必要的影響

## Hyperparameter
### Describe your hyperparameter you use and how you decide it
```
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --summary_column title \
    --source_prefix "summarize: " \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --num_train_epochs 15\
    --checkpointing_steps epoch\
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_source_length 2048 \
    --output_dir ./output_mt5_all_train2 \
    --report_to wandb \
    --num_beams 4 \
    --with_tracking \
```
- Max source length, Max target length: Max source length設為2048，Max target length為預設的128，因分析了train data的長度
```
最大 title 長度: 105
最大 maintext 長度: 30435
平均 title 長度: 25.68825426070935
平均 maintext 長度: 1436.1695532012898
```
- Learning rate: 選擇了較小的學習率（2e-5），因為發現在learning過高的情況下，loss上下震盪的情況較明顯，最後結果也不好。
- --lr_scheduler_type linear :在使用線性調度時，學習率會線性遞減，從初始學習率（如 2e-5）開始，隨著訓練的進展逐步減小，直到訓練結束。線性學習率調度在訓練的後期提供更穩定的小步調學習，減少模型過度更新的風險，從而有助於提升模型的最終性能。
- gradient_accumulation_steps, batch size: 使用effective batch size為4，因為在嘗試過程中發現effective batch size為16時反而效果較差
- num_beams 4: 因為chatgpt推薦，使用後發現效果不錯
## Learning Curves
以所有的training data做訓練，y軸為public的rouge score*100後
![W&B Chart 2024_10_14 下午3_56_27](https://hackmd.io/_uploads/rkY6Rrckyl.png)
![W&B Chart 2024_10_14 下午3_56_11](https://hackmd.io/_uploads/rJYaArqkJl.png)
![W&B Chart 2024_10_14 下午3_56_03](https://hackmd.io/_uploads/ryKa0Hc1kg.png)
![W&B Chart 2024_10_14 下午3_55_52](https://hackmd.io/_uploads/BytpArqy1g.png)

## Generation Strategies
### Describe the detail of the following generation strategies:
- Greedy: 在每一步生成詞彙（token）時，直接選擇模型當前認為機率最高的詞彙，完全不考慮後續可能的選擇或多種候選方案。可能錯過能提升整體文本品質的更佳選擇，難以到達global optima，但計算快速。

- Beam Search: 在每一步選擇多個具有較高機率的詞彙，並同時保留多個候選序列(beams)，num_beams 決定了保留的數量。最終會選擇整體機率最高的beam作為輸出。當beams過小時就接近greedy，beams過大時除了計算成本上升外，也會導致生成general的結果。

- Top-k Sampling: 在每一步生成時，模型不僅選擇機率最高的詞彙，而是從機率前 k 個詞彙中隨機選擇一個，這樣可以在生成過程中引入隨機性和多樣性。如果 k 設置過大，可能會選擇機率很低的詞彙，導致不連貫或無意義的生成；設置過小，則隨機性不足。

- Top-p Sampling: Top-p 取樣根據詞彙的累積機率來決定可選範圍。在每一步，模型會從累積機率超過 p 的詞彙中隨機選擇一個。生成的文本更平滑自然，因為它會自動調整可選詞彙的數量，而不僅僅根據固定的 k 值選擇詞彙。

- Temperature: 進入decoding algorithm前，計算softmax時，先除一個temperature hyperparameter。當 temperature 越低時，模型更加保守，傾向於選擇高機率的詞彙；當 temperature 越高時，模型更加隨機，可能選擇低機率的詞彙，生成更具多樣性和創造性的文本。
	- temperature = 1.0：使用標準的機率分佈。
	- temperature < 1.0：減少隨機性，模型生成更為穩定和保守的結果。
	- temperature > 1.0：增加隨機性，模型生成更為多樣化的結果。

### Hyperparameters
#### Try at least 2 settings of each strategies and compare the result.
![image](https://hackmd.io/_uploads/rJgCF46yJl.png)
- **Beam Search**: 當num_beams為5時，三個分數皆大於beams 4和6，因此我在參數中加入do_sample=True(第二筆num_beams = 5)，造成生成分數下降
	- 因此除了greedy，以下num_beams都設成5
- **greedy**: ROUGE-1 和 ROUGE-2 的分數相比於 beam search 明顯較低，這說明 greedy search 相對於 beam search 更加單調，不能產生多樣性更高的摘要。
- **temperature**: 當溫度為 0.1 時，ROUGE-1 和 ROUGE-L 的分數為 0.273264 和 0.241779，接近於 beam search，顯示較低溫度能夠生成接近最佳的摘要。當溫度提高到 2.0 時，分數下降明顯，ROUGE-1 降到 0.245816，說明過高的隨機性使生成質量下降。
- **Top-k**:  k 為 50 時，ROUGE-1 分數最好，k為10時ROUGE-2、ROUGE-L分數最好
- **Top-P**: 0.50在ROUGE評分上表現最佳

#### What is your final generation strategy?
num_beams為5，max_length=128，生成後將標點符號半形轉成全形