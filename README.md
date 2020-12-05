# VMSMO
Official code and dataset link for ''VMSMO: Learning to Generate Multimodal Summary for Video-based News Articles''

## About the corpus
VMSMO corpus consists of 184,920 document-summary pairs, with 180,000 training pairs, 2,460 validation and test pairs.

We first publish the link (https://drive.google.com/drive/folders/1MpVv9naDaLINIo4ZKjGoZZHqp7v3_b-A?usp=sharing) to download each case in the dataset. The dataset consists of train.json, valid.json, and test.json. In each item in the json file, there are: 
ID: the ID number of the news
content: the content of news
original_pictures: whether the original microblog has pictures
video_uRL: video URL
image_uRL: video cover image URL
publish_place: the place of publication
publish_time: the release time of microblog
publish_tool: microblog publishing method
Up_num: number of likes
retweet_num: number of forwarding
comment_num: number of comments
title: title of the weibo

The complete dataset is coming soon.

## About the code

### Requirements
<ul>
<li> python = 3.6
<li> tensorflow = 1.9
<li> numpy = 4.2
<li> opencv python = 1.16
</ul>

### Commands
train:
```python
python run_summarization.py --mode=train --data_path=* --test_path=* --vocab_path=* --log_root=logs --exp_name=vmsmo --max_enc_steps=100 --max_dec_stpes=30 --vocab_size=50000 --lr=0.001
```

test:
```python
python run_summarization.py --mode=decode --data_path=* --test_path=* --vocab_path=* --log_root=logs --exp_name=vmsmo --max_enc_steps=100 --max_dec_stpes=30 --vocab_size=50000 --lr=0.001
```


## Citation
We appreciate your citation if you find our dataset and code beneficial.

```
@inproceedings{Li2020VMSMO,
  title={VMSMO: Learning to Generate Multimodal Summary for Video-based News Articles},
  author={Mingzhe Li, Xiuying Chen, Shen Gao, Zhangming Chan, Dongyan Zhao, and Rui Yan},
  booktitle = {EMNLP},
  year = {2020}
}
```
