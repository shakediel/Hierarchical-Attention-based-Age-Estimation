# Hierarchical-Attention-based-Age-Estimation

this is the officiail implementation for [ Hierarchical Attention-based Age Estimation and Bias Estimation](https://arxiv.org/abs/2103.09882).

1. dowload the relevant dataset:
https://biu365-my.sharepoint.com/:u:/g/personal/kellery1_biu_ac_il/EUK1JbQuqUtLvqyGRNJQzpcBUowyIGHinwulz9Xp99w10A?e=tVxFee


in order to reproduce the article results:
1. dowload the relevant dataset
2. run recognition_main.py in order to pretrain the selected model
3. run the age estimator training, pretrained for recognition
   1. in order to train a cnn based model, run unified_main.py
   2. in order to train the transformer based model, run transformer_main.py






