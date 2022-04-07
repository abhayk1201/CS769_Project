* Used code from CS769 Assignment 2 for baseline implementation: [Link](https://github.com/JunjieHu/cs769-assignments/tree/main/assignment2)


### Reference: 
This is an exercise in developing a minimalist version of BERT, adapted from CMU's [CS11-747: Neural Networks for NLP](http://www.phontron.com/class/nn4nlp2020/) by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt and Brendon Boldt

[Vaswani el at. + 2017] Attention is all you need https://arxiv.org/pdf/1706.03762.pdf

# Code Structure

## bert.py
This file contains the BERT Model whose backbone is the [transformer](https://arxiv.org/pdf/1706.03762.pdf). We recommend walking through Section 3 of the paper to understand each component of the transformer. 

### BertSelfAttention
The multi-head attention layer of the transformer. This layer maps a query and a set of key-value pairs to an output. The output is calculated as the weighted sum of the values, where the weight of each value is computed by a function that takes the query and the corresponding key. To implement this layer, you can:
1. linearly project the queries, keys, and values with their corresponding linear layers
2. split the vectors for multi-head attention
3. follow the equation to compute the attended output of each head
4. concatenate multi-head attention outputs to recover the original shape

<img src="https://render.githubusercontent.com/render/math?math=Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}}V)">

### BertLayer
This corresponds to one transformer layer which has 
1. a multi-head attention layer
2. add-norm layer
3. a feed-forward layer
4. another add-norm layer

### BertModel
This is the BertModel that takes the input ids and returns the contextualized representation for each word. The structure of the ```BertModel``` is:
1. an embedding layer that consists of word embedding ```word_embedding``` and positional embedding```pos_embedding```.
2. bert encoder layer which is a stack of ```config.num_hidden_layers``` ```BertLayer```
3. a projection layer for [CLS] token which is often used for classification tasks

The desired outputs are
1. ```last_hidden_state```: the contextualized embedding for each word of the sentence, taken from the last BertLayer (i.e. the output of the bert encoder)
2. ```pooler_output```: the [CLS] token embedding

The detailed model descriptions can be found in their corresponding code blocks
* ```bert.BertSelfAttention.attention``` 
* ```bert.BertLayer.add_norm```
* ```bert.BertLayer.forward```
* ```bert.BertModel.embed```


## classifier.py
This file contains the pipeline to 
* call the BERT model to encode the sentences for their contextualized representations
* feed in the encoded representations for the sentence classification task
* fine-tune the Bert model on the downstream tasks (e.g. sentence classification)


### BertSentClassifier
This class is used to
* encode the sentences using BERT to obtain the pooled output representation of the sentence.
* classify the sentence by applying dropout to the pooled-output and project it using a linear layer.
* adjust the model paramters depending on whether we are pre-training or fine-tuning BERT

## optimizer.py 
This is where `AdamW` is defined.
You will need to update the `step()` function based on [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) and [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
There are a few slight variations on AdamW, pleae note the following:
- The reference uses the "efficient" method of computing the bias correction mentioned at the end of "2 Algorithm" in Kigma & Ba (2014) in place of the intermediate m hat and v hat method.
- The learning rate is incorporated into the weight decay update (unlike Loshchiloc & Hutter (2017)).
- There is no learning rate schedule.



## base_bert.py
This is the base class for the BertModel. It contains functions to 
1. initialize the weights ``init_weights``, ``_init_weights``
2. restore pre-trained weights ``from_pretrained``. Since we are using the weights from HuggingFace, we are doing a few mappings to match the parameter names


## tokenier.py
This is where `BertTokenizer` is defined. 

## config.py
This is where the configuration class is defined.

## utils.py
This file contains utility functions for various purpose.



### Acknowledgement
Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
