# Requirenments
transformers>=3.0.2
torch>=1.7.0
tqdm
joblib
csv
thop

# Datasets Download
ReCO : https://drive.google.com/drive/folders/1rOAoKcLhMhge9uVQFM2_D1EU0AjnpWFa?usp=sharing

RACE: http://www.cs.cmu.edu/~glai1/data/race/

our model is also available on some classification task.

LCQMC:http://icrc.hitsz.edu.cn/Article/show/171.html

Ag.news and Dbpedia: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

Book Review: https://github.com/autoliuweijie/FastBERT/tree/master/datasets/douban_book_review


# RACE preprocess
save the data in a josn file
```     bash
python format_source.py ./data/RACE/train
python format_source.py ./data/RACE/test
python format_source.py ./data/RACE/dev

```

# Train


```	bash
data_set_type='ReCO'
model_type='bert-base-chinese'
alpha=5
layers=3
batch_size=6
epoch=20
total_batch_size=48

data_path="./data/$data_set_type"
output_dir="./${data_set_type}_${model_type}_alpha${alpha}_bth${batch_size}_${total_batch_size}_layer${layers}"
mkdir -p $output_dir

python3 -u train.py \
    --model_type $model_type \
    --output_dir $output_dir \
    --batch_size $batch_size\
    --epoch $epoch\
    --gradient_accumulation_steps $((${total_batch_size}/$batch_size)) \
    --data_path $data_path \
    --alpha $alpha \
    --lr 3e-5 \
    --max_grad_norm 2.0 \
    --split_layer $layers \
    --warmup_proportion 0.1 \
    1>$output_dir/log.txt 2>&1
```

# Test
## Performance first 
```bash
python3 test_fast.py \
    --model_type $model_type \
    --output_dir $output_dir \
    --batch_size 16 \
    --data_path $data_path \
    --alpha $alpha \
    --split_layers $layers \
    --speed 0
```





## Speed first
```bash
python3 test_fast.py \
    --model_type $model_type \
    --output_dir $output_dir \
    --batch_size 16 \
    --data_path $data_path \
    --split_layers $layers \
    --speed 0.1
```


# Similarity
```bash
python3 test_sim.py \
    --model_type $model_type \
    --output_dir $output_dir \
    --batch_size 16 \
    --data_path $data_path \
    --split_layers $layers
```
	
