# CS769_Project
## BERT-MTL: Multi-Task Learning Paradigm for Improved Emotion Classification using BERT Model


Course Project for [CS769: Advanced Natural Language Processing](https://junjiehu.github.io/cs769-spring22/)

Team Members: [Abhay Kumar](https://abhayk1201.github.io/), Neal B Desai, Priyavarshini Murugan

**Report:** [Read-only Overleaf link](https://www.overleaf.com/read/dcxyxvjqkjsh)

* [Pdf report](./report.pdf)

### Install
If running on Google Colab, just install the python packages using
`pip install -r requirements.txt`

Pre-requisites: 
python 3.7
pytorch
`setup.sh` will install the `goemotion` environment with required python packages.



## Goemotion Data
**GoEmotions** is a corpus extracted from Reddit with human annotations to 28 emotion labels (27 emotion categories + Neutral). 
* Processed data is already uploaded to `data` folder in this repository for convenience, you can skip the following steps.
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

### Geomotion Data Hierarchial grouping
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

### Single Task (Goemotion) Running instructions
Change corresponding `Config`/{}.json file for the required taxonomy and pass the grouping/taxonomy as an argument like.
You can set `do_train`, `do_eval` depending on whether you want training, evaluation or both. You can also change different hyperparameters like `train_batch_size`, `learning_rate`, `num_train_epochs` etc.

```bash
$ python3 goemotions_classifier.py --taxonomy {$TAXONOMY}

$ python3 goemotions_classifier.py --taxonomy original
$ python3 goemotions_classifier.py --taxonomy sentiment
$ python3 goemotions_classifier.py --taxonomy ekman
```

## Sentiment140 Data
* [Original Dataset download link](http://help.sentiment140.com/for-students)
* We have shared the [processed dataset drive link](https://drive.google.com/drive/folders/1jUhA1NNYFo8dhfp1l66oDuPPlPGrx-qj?usp=sharing)
* Train set: Total of 1,600,000 training tweets (800,000 tweets with positive sentiment, and 800,000 tweets with negative sentiment).
* Test set: Composed of 177 negative sentiment tweets and 182 positive sentiment tweets.


## Suicide and Depression Detection Data
* [Dataset download link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

## Multi-task Learning Running instructions


**Setup**: Download the pretrained BERT pytorch model from [google drive link](https://drive.google.com/drive/folders/1xPDf-ZCNJG96-awfiYbU_nS5PCA3dj2r?usp=sharing). You should skip the next step if you download this pytorch pre-trained model. We have already shared google drive link after doing the conversion.

*Otherwise*, You can convert TensorFlow checkpoint for BERT ([pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`convert_tf_checkpoint_to_pytorch.py`](./scripts/convert_tf_checkpoint_to_pytorch.py) script.


If you are running in google colb, modify your path variables as per your setup and run the following code.

```bash
python run_multi_task.py \
  --seed 42 \
  --output_dir /content/drive/MyDrive/769/assignment3/Bert-Multi-Task2/Tmp_Model/MTL \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /content/drive/MyDrive/769/assignment3/Bert-Multi-Task2/data/ \
  --vocab_file /content/drive/MyDrive/769/assignment3/Bert-Multi-Task2/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file /content/drive/MyDrive/769/assignment3/Bert-Multi-Task2/config/pals_config.json \
  --init_checkpoint /content/drive/MyDrive/769/assignment3/Bert-Multi-Task2/uncased_L-12_H-768_A-12/pytorch_model.bin \
  --max_seq_length 50 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1
```

Note the different arguments:
* tasks: `all` will run for all tasks together.  'single' will run for a single task (you need to pass another arg --task_id for the required task)
* sample: Different sampling schemes for training different tasks.  
  * `rr`: Round Robin: Select a batch of training examples from each task, cycling through them in a fixed order. However, this may not work well if differnt tasks have different numbers of training examples.
  * `prop`: Select a batch of examples from task i with probability <img src="https://render.githubusercontent.com/render/math?math=p_i"> at each training step, where <img src="https://render.githubusercontent.com/render/math?math=p_i">  proportional to <img src="https://render.githubusercontent.com/render/math?math=N_i">, the
training dataset size for  <img src="https://render.githubusercontent.com/render/math?math=i^{th}"> task.
  * `sqrt`: Similar as `prop`, but sampling is proportional to the square root of the training dataset size.
  * `anneal`: *annealed sampling* method changes the proportion with each epoch so as to become more more equally towards the end of training (where we are most concerned about interference). 
* max_seq_length: The maximum total input sequence length after WordPiece tokenization. Otherwise, it will be trucated or padded.
* train_batch_size:  Batch size for training.
* num_train_epochs:  Number of epochs of training.
* do_train:  If you want to run training.
* do_eval: Whether to run eval on the dev set
* data_dir: Directory path which contains different task dataset.
* vocab_file: Directory path of pretrained model downloaded from google drive link in the setup above.
* init_checkpoint: Directory path of pretrained model downloaded from google drive link in the setup above.
* bert_config_file: Differnt configs for Multi Task Learning settings.


## References
[Awesome Multi-Task Learning](https://github.com/Manchery/awesome-multi-task-learning)

[BERT and PALs: Projected Attention Layers for
Efficient Adaptation in Multi-Task Learning](https://github.com/AsaCooperStickland/Bert-n-Pals)

[Goemotion Google Data and Baseline Model](https://github.com/google-research/google-research/tree/master/goemotions)
