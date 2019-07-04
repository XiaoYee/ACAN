# ACAN
Code for NAACL 2019 paper: "Adversarial Category Alignment Network for Cross-domain Sentiment Classification" [(pdf)](https://www.aclweb.org/anthology/N19-1258)

## Dataset & pretrained word embeddings
You can download the datasets (amazon-benchmark) at [[Download]](https://drive.google.com/open?id=1rEwGXdEqt2xZwtJi7RHeRcSZDuYrSIpq). The zip file should be decompressed and put in the root directory.

Download the pretrained Glove vectors [[glove.840B.300d.zip]](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip). Decompress the zip file and put the txt file in the root directory.

## Train & evaluation
You can find arguments and hyper-parameters defined in train_batch.py with default values.

Under code/, use the following command for training any source-target pair from the amazon benchmark:
```
CUDA_VISIBLE_DEVICES="0" python train_batchs.py \
--emb ../glove.840B.300d.txt \
--dataset amazon \
--source $source \
--target $target \
--n-class 2  \
--lamda1 -0.1 --lamda2 0.1 --lamda3 5 --lamda4 1.5 \
--epochs 30 
```
where *--emb* is the path to the pre-trained word embeddings. *$source* and *$target* are domains from the amazon benchmark, both in ['book', 'dvd', 'electronics', 'kitchen']. --n-class denoting the number of output classes is set to 2 as we only consider binary classification (positive or negative) on this dataset. All other hyper-parameters are left as their defaults.

## Dependencies
The code was only tested under the environment below:
* Python 2.7
* Keras 2.1.2
* tensorflow 1.4.1
* numpy 1.13.3

## Cite
If you use the code, please cite the following paper:
```
@InProceedings{qu-etal-2019-adversarial,
  author    = {Qu, Xiaoye and Zou, Zhikang and Cheng, Yu and Yang, Yang and Zhou, Pan},
  title     = {Adversarial Category Alignment Network for Cross-domain Sentiment Classification},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
  publisher = {Association for Computational Linguistics}
}
```
