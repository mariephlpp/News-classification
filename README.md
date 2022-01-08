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

```
pred = model.predict(X_test)
```
```
predictions = []

for i in pred:
    if i >= 0.5:
        predictions.append(1)
    else:
        predictions.append(0) 
```
* Accuracy on the test set: 0.989
* Precision on the test set: 0.985
* Recall on the test set: 0.994

![Unknown](https://user-images.githubusercontent.com/87387511/148624705-c33f5082-7e16-41b0-8001-09d8552cd216.png)

![Unknown-2](https://user-images.githubusercontent.com/87387511/148624703-f4aaa6bb-86f8-4efc-818f-cd6394921436.png)

The results are very promising that this model would work well for another text classification. 

### LSTM
### BERT

* What is an BERT?

The specificity of BERT models is they use the transformer encoder architecture to process each input text token in the full context of all tokens before and after. Several BERT models exists. In this project we used the bert base model, which consists of 12 layers of transformer encoder, 12 attention heads, 768 hidden size, and 110M parameters.

* How does it work? 

The transformer is the part of the model that enables BERT tooutperform other models. The transformer processes a given word in relation to all other words in the sentence, rather than processing them one by one. It allowes to understand fully the context of the word. 

BERT models are pre-trained on a large corpus of text (English Wikipedia and Books Corpus)  and then refined for specific tasks. Here, we used pre-trained BERT model from Hugging Face. BERT is bidirectional and pre-trained on two tasks. The first one is Masked Language Model: the model is trained by hiding a word in a sentence and then trying to predict it based on the masked word's context. The second one is Next Sentence Prediction: this time the model is training by looking at two sentences and predicts if they have a sequential connection or just a random one. 

It is necessary to train the BERT model on a new task to specialize the model, this is called finetuning. Indeed, BERT is pre-trained using only an unlabeled text corpus, thus additional layers of neurons need to be added to transform the model into a classifier example.

An attention process is used, it means that each output element is connected to every input element, and the weights between them are dynamically calculated based upon their connection. 





* Issues with BERT 





## Conclusion


