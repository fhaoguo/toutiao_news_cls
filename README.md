# 头条新闻分类任务
本项目方案适合新闻短文本分类，可以在此基础上进一步完善实现其他短文本分类任务，欢迎fork 和 star。数据及模型因存储空间限制请自行搜索获取。

## 分类效果对比
|                   | f1     | precision | recall | accuracy |
|--------------------------|--------|--------|--------|--------|
| RandomForest(epoches=10) | 0.4773 | 0.4980 | 0.4693 | 0.5048 |
| TextCNN(epoches=2)       | 0.5188 | 0.5395 | 0.5077 | 0.5266 |
| Bert(epoches=2)          | 0.5709 | 0.5698 | 0.5762 | 0.5690 |
| BertWithHead(epoches=2)  | 0.5536 | 0.5815 | 0.5514 | 0.5654 |
| NeZhaWithHead(epoches=2) | 0.5524 | 0.5774 | 0.5439 | 0.5622 |

## 数据分析及预处理模块
```
Analysis.ipynb
```

## RandomForest 分类baseline搭建
```
most_simple_way.py
```

## TextCNN 模型分类实现
详情点击[text_cnn](./text_cnn)

## Bert 模型分类实现
#### bert 版本
+ 详情点击[bert_base](./bert_base)
#### bert with head 版本
+ 详情点击[bert_head_base](./bert_head_base)

## NeZha 模型分类实现
详情点击[nezha_head_base](./nezha_head_base)

## Focal Loss优化版本
详情点击[nezha_head_focalloss](./nezha_head_focalloss)

## FGM PGD对抗训练版本
详情点击[nezha_head_fl_fgm_pgd](./nezha_head_fl_fgm_pgd)
