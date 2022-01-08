# Deep learning project

## Marie PHILIPPE & Claire SERRAZ - M2 D3S

![alt text](https://www.tse-fr.eu/sites/all/themes/tse/images/site/logo-tse.png)

## Table of contents
* [Introduction](#introduction)
* [Literature](#literature)
* [Data](#data)
* [Data cleaning](#data-cleaning)
* [Models](#models)
	* [LSVM](#lsvm)
	* [RNN](#rnn)
	* [LSTM](#lstm)
	* [BERT](#bert)
* [Conclusion](#conclusion)

## Introduction


## Literature

## Data

## Data cleaning 

## Models
### LSVM
### RNN

* What is an RNN? 

Deep neural networks usually assume independence between input and outputs. However, it isn't the case for RNNs. Indeed, RNNs have an internal memory thanks to a hidden state feature. It means that information are take from previous inputs to influence the next input and output within each sequence. RNN are therefore known to understand better sequences and their context. 

Here, we are in the case of a "many to one" application. Indeed, we have several inputs (group of words) but only one input (the type of news). 

* How does it work? 

Independent activations are changed to dependent activations by choosing the same weights and biases to all the hidden layers. It enables to reduce the complexity when increasing the parameters and memorizing. The layers are joined so the weights and biases of the layers are the same, which then gives a single recurrent layer.

* Steps 

The first step is to tokenize the text. The text corpus is vectorized: each text is turned into a sequence of integers. 
20 000 is the maximum number of words that is kept.
The tokenizer is created then the internal vocabulary based on a list of texts is updated.

```
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
```

The list of sequences is then transformed to a numpy array. 

```
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=256)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=256)
X_valid = tf.keras.preprocessing.sequence.pad_sequences(X_valid, maxlen=256)
```

The second step is to create the sequential model which is the following. 

```
model = Sequential()

model.add(Embedding(20000, 128))
model.add(Bidirectional(LSTM(25, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
```

The first hidden layer is the embedding. It is initialized with random weights. The embedding for all of the words in the training dataset will be learned. 
It has two argument: the size of the vocabulary in the text data (input_dim=20000) and the size of the vector space in which words will be embedded (output_dim=128).    
The second layer is a stacked bidirectional LSTM. The LSTM algorithm will be explained in the next sub-section.  
GlobalMaxPool1D downsamples the input representation by taking the maximum value.  
Among the next layers are Dropout layers. There are used as a regulazition technique. The principle in to shut some neurons down so they are less sensitive to the activation of another neuron.      
Each neuron in the dense layer receives input from all neurons of the previous layer and an activation function, here relu, can be associated to it.

One all the previous steps where done we could configure our model with a loss, a metric and an optimizer. 
```
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
```

The binary crossentropy loss was chosen since the news can be only of two differnt types. As an optimizer was chosen ADAM (Adaptive Moment Estimation) which combines momentum and RMSprop. For the metrics, the accuracy is used. 

* Result

Once all the previous steps done we could validate the model and then predict the class of the news. 

We had to fit the model. 
```
model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), batch_size=32)
```
The following results enabled us to validate the model:

* Train loss: 0.009
* Train accuracy: 0.998

* Test loss: 0.025
* Test accuracy: 0.992

Fitting again the model with the test set instead of the validation set and prediciting the classes we got:

* Accuracy on the tes set: 0.9897550111358575
* Precision on the test set: 0.985040797824116
* Recall on the test set: 0.994053064958829


### LSTM
### BERT

## Conclusion


