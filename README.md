# NER Using Bidirectional LSTM

## Idea
NER tasks in the past have been largely engineered by domain experts to model language specific features. This becomes a challenge when the annotated dataset is limited. Recent developments in the space of Artificial Neural networks have shown good results in NLP tasks, without extensive language modeling. The idea is to use semantic information learnt from large unannotated corpuses(which is available more easily), and then use this information, which is captured in the form of embedding. This is then used for the NLP tasks to train specific tasks under a supervised setting for which a limited annotated dataset is obtained. Extensive research has been done to show good improvements in benchmark performance with this approach. In this project we will explore this idea by following some of the research papers and further experiment by implementing such a system.

## Approach:
Built a Neural network Model which is based on Bi-Directional Recurrent Neural Network based Long Short Term Memory network, for Named Entity Recognition(NER). 
The model uses the Pre-trained word embedding word2vec trained on Google news which are learnt by an unsupervised learning algorithm on an unannotated corpus.  Additional syntactic features are added to help improve the performance.
We have used TensorFLow to implement the algorithm. 
The model achieves good performance on the CoNNL2003 NER task, as observed on the testsets.

## Features:
We are first processing the raw sentences by representing them as a variable sequence of words, where each word is represented as a feature vector of word embeddings from word2vec of 300 dimensions. This is then combined with additional feature elements such as Part-of-Speech Tag, Syntactic Chunk Tag (available in the corpus) and Initial letter Capital Case (if the word is capitalized on the initial letter, this can help with the entity dectection). With these transformations, each word is represented as a 311 dimensional vector. 
Also, we have filtered the train, test and validation datasets to filter out sentences that are larger than 30 words. This is to keep the training and test cycles shorter and we have seen adequate representation of the data in sentences less than or equal to length 30.

## Neural Network Architecture:
The Bidirectional Neural network was built with 256 hidden units, with one hidden layer of Forward and Backward LSTM Cells, and then combined with Softmax output layer.


## Corpus
The NER model was trained and tested with the CoNLL 2003 NER Task English dataset. The dataset contains the newswire articles annotated for NER with four entity types: Person (PER), Location(LOC), Organization(ORG), Miscellaneous (MISC) along with non entity elements tagged as (O). The corpus consists of separate training, validation and test sets. 
We have used the training set to train the model, and tuned the model using the validation set. The generalization accuracies are reported on the test set. 
We are using Precision, Recall and F1-score on each of the Entity Types, and also on an overall level, including all the Four types.


