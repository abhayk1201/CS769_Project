# CS769_Project
Course Project for [CS769: Advanced Natural Language Processing](https://junjiehu.github.io/cs769-spring22/)

Team Members: [Abhay Kumar](https://abhayk1201.github.io/), Neal B Desai, Priyavarshini Murugan

## Install
If running on Google Colab, just install the python packages using
`pip install -r requirements.txt`

Pre-requisites: 
python 3.7
pytorch
`setup.sh` will install the `goemotion` environment with required python packages.



## Goemotion Data
**GoEmotions** is a corpus extracted from Reddit with human annotations to 28 emotion labels (27 emotion categories + Neutral). 
* [Goemotion Dataset Link](https://github.com/google-research/google-research/tree/master/goemotions/data)
* Dataset splits: training dataset (43,410), test dataset (5,427), and validation dataset (5,426).
* Maximum sequence length in training and evaluation datasets: 30
* The emotion categories are: _admiration, amusement, anger, annoyance, approval,
caring, confusion, curiosity, desire, disappointment, disapproval, disgust,
embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness,
optimism, pride, realization, relief, remorse, sadness, surprise_.
* Raw dataset can be downloaded using
```
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

## Geomotion Data Hierarchial grouping
 * **Original GoEmotions** (27 emotions + neutral)
 * **Sentiment Grouping** (positive, negative, ambiguous + neutral)
 * **Ekman** (anger, disgust, fear, joy, sadness, surprise + neutral), where
 **anger** : *anger, annoyance, disapproval*, 
 **disgust** : *disgust*,
 **fear** : *fear, nervousness*, 
 **joy** : *all positive emotions*, 
 **sadness** : *sadness, disappointment, embarrassment, grief, remorse* 
 **surprise** : *all ambiguous emotions*

`Config` directory has the respecttive config files for above groupings/taxonomy.

## Running instructions
Change corresponding `Config`/{}.json file for the required taxonomy and pass the grouping/taxonomy as an argument like.
You can set `do_train`, `do_eval` depending on whether you want training, evaluation or both. You can also change different hyperparameters like `train_batch_size`, `learning_rate`, `num_train_epochs` etc.

```bash
$ python3 goemotions_classifier.py --taxonomy {$TAXONOMY}

$ python3 goemotions_classifier.py --taxonomy original
$ python3 goemotions_classifier.py --taxonomy sentiment
$ python3 goemotions_classifier.py --taxonomy ekman
```


## References
[Awesome Multi-Task Learning](https://github.com/Manchery/awesome-multi-task-learning)

[Goemotion Google Data and Baseline Model](https://github.com/google-research/google-research/tree/master/goemotions)
