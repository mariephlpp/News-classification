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

Text classification is a popular task in Natural Language Processing (NLP). The purpose is to classify text based on pre-defined classes, usually sentiments or topics. Text can contain a lot of information, however extracting this information can take time, especially due to its unstructured nature. With nowadays's deep learning models, classifying text is getting easier. Fake news are spreading false information in order to influence readers' beliefs which usually damages a person's, a company's... reputation. Thus, they are a real issue today, especially since a vast volume of text data is generated everyday in the form of social media, websites, news... etc. 

This work focuses on article news taken from the *Fake and real news dataset* from kaggle. The aim is to classify these news into two different categories: true or fake. To do so, four different models are used:
* Linear Support Vector Machine (LSVM)       
* Recurrent Neural Networks (RNN)    
* Long Short-Term Memory (LSTM)    
* Bidirectional Encoder Representations from Transformers (BERT). 

First, we will explain quickly what has already been done in terms of news classification, then introduce our data and its cleaning and finally we will present our models and the results we got. 

## Literature

For this project we mainly used two papers:
- A survey on Natural Language Processing for Fake News Detection by Oshikawa et al. (2020)
- Automatic Detection of Fake News by Pérez et al. (2018)

The first paper details the challenges, tasks, and NLP solutions to answers to the challenge of fake news detection. They first transformed the problem into a binary classification problem (fake - real), but then added other classes for not completely real and not completely fake news. According to them, the following model can be used for text classification: 
- Support Vector Machine,
- Naive Bayes Classifier, 
- Logistic Regression,
- Decision Tree,
- Random Forest Classifier,
- Recurrent Neural Network (RNN), especially the Long Short-Term Memore(LSTM),
- Convolutional Neural Network (CNN),
- Multi Source Multi-class Fake News Detection framework (MMFD).

The results are compared on three different datasets. For the first dataset, it appears the Deep CNN and a model called NLP shallow gives the best accuracy scores (respectively 96.2% and 92.1%) when for the other models tested, the accuracy scores remain below 50%. For the second dataset, the results are slightly higher on average than the ones described before, the best accuracy score is 68% and 68,4% for two LSTMs. For the third dataset, the results are way better, all above 50%. The models used are also different from the ones described above. The best model would be a GCN, and a HC-CB-3.

The second paper also covers the processus of text classification over two different datasets. They use a LSVM Classifier only but play with the data cleaning, on several options, and on the several features:
- Punctuation: Twelve types of punctuaction derived from the Linguistic Inquiry and Word COunt software (LIWC) are used.
- LIWC: It is a lexicon that allows to extract the proportions of words into several ategories. It can represents psycholinguisitic processes, summary categories, part-of-speech categories...
- Readability: Those are features to indicate text understandability (number of characters, complex words...)
- Ngrams: It aims to extract unigrams and bigrams derived from the bag of words representatio of each news articles. These features are then tf-idf vectorized. 
- CFG (Context free grammars): It is a tree that is composed of all the lexicalized production rules combined with their ancient nodes. These features are also tf-idf vectorized.
- All features.

For the first dataset, the best results is when only using the readability, and then the accuracy score is 78%. When using all the features, we get an accuracy score of 74%. The opposite is seen on the second dataset: the accuracy score when only using the readability is the worst (62%). The best accuracy scores is when using all features (76%) and when using the LIWC lexicon (74%). This paper clearly shows the importance of the choice of dataset for the classification problem. 

## Data

The data used come from kaggle: *Fake and real news dataset* (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It contains a first dataset with 23481 fake news and a second one with 21417 true news. Each of the datasets has 4 columns: the title, the text, the subject and the date. In our models we will mainly use the text and sometimes the title. To be able to use the data, it had to be cleaned. 

## Data cleaning 

We had two datasets but needed just one, thus the first thing we did was to merge the fake and true news datasets and specified each time the class (true or fake) of the news. Once this step was done, we could clean the title and text using the Natural Language Toolkit.  

We created several functions:
- Convert_text_to_lowercase: It allows to put all the charactes in lowercase.
- Remove punctuation: It allows to remove some useless punctuation.
- Tokenize_sentence: It allows to tokenize the sentence, i.e., to split the word one
by one as vectors of letters.
- Remove_stop_words: It allows to remove some basic english words such as "the", "a", "an"...
- Lemm: It keeps only the roots of words. 
- Reverse_tokenize_sentence: It allows to end the tokenization we have dropped
some useless words.

 ```
# Convert text to lowercase
def convert_text_to_lowercase(df, colname):
    df[colname] = df[colname].str.lower()
    return df

def not_regex(pattern):
        return r"((?!{}).)".format(pattern)

# Remove punctuation and new line characters '\n'
def remove_punctuation(df, colname):
    df[colname] = df[colname].str.replace('\n', ' ')
    df[colname] = df[colname].str.replace('\r', ' ')
    alphanumeric_characters_extended = '(\\b[-/]\\b|[a-zA-Z0-9])'
    df[colname] = df[colname].str.replace(not_regex(alphanumeric_characters_extended), ' ')
    return df

# Tokenize sentences
def tokenize_sentence(df, colname):
    df[colname] = df[colname].str.split()
    return df

# Remove the stopwords
def remove_stop_words(df, colname):
    df[colname] = df[colname].apply(lambda x: [word for word in x if word not in sw])
    return df

# Lemmatisation (get the root of words)
def lemm(df, colname):
    df[colname] = df[colname].apply(lambda x: [wnl.lemmatize(word) for word in x])
    return df

# Convert tokenized text to text
def reverse_tokenize_sentence(df, colname):
    df[colname] = df[colname].map(lambda word: ' '.join(word))
    return df

# Apply all the functions the text
def text_cleaning(df, colname):
    df = (
        df
        .pipe(convert_text_to_lowercase, colname)
        .pipe(remove_punctuation, colname)
        .pipe(tokenize_sentence, colname)
        .pipe(remove_stop_words, colname)
        .pipe(lemm, colname)
        .pipe(reverse_tokenize_sentence, colname)
    )
    return df
```

A new dataframe was created that we exported as a csv. This step enabled us to not run the cleaning each time. 

## Models

From the literature review, we decided to go on with the LSVC model. Then, we tried Recurrent Neural Networks: a simple one, described in the literature  as well, and the Long Short Term Memory, that we saw in class and that is also commonly used for text classification. Finally, we decided to try to implement a Bidirectional Encoder Representations from Transformers (BERT) model.

### LSVC

* What is a LSVC?

It is a linear Support Vector Machine (SVM), that is a supervised algorithm that balance power and flexibility. It appeared in the 60s but were well defined in the 90s. A LSVC fixes the kernel set to linear. This way, it has more flexibiity to choose the penalties and loss functions. 

* Some vocabulary

-An hyperplane is a decision space divided between a set of points having different classes.

-A margin is the gap between two lines on the closest data points on different classes. The largest margins are the best.

-A Support Vector is the space of datapoints that are the closest to the previoulys defined hyperplane.

* Tf-idf vectorization

To use the LSVC model, we need to apply the Term Frequency - Inverse Document Frequency vectorization on the text variable. It is a weighting method that help to estimate the lexical relevance of a word contained in a document, relative to the corpus. The weight proportionally increases according to the number of occurences of the word in the given document, but also varies according to the frequency of the word in the corpus. Therefore, here is applied a relation between a document and a set of documents that share similarities of lexical fields. 

For instance, in the case of a query containing the term X? A document is considered more relevant as a response if it has a certain occurrence of the word X, and if X has a rarity in other documents related to the first.

The term frequency is the frequency of x in y, and the inverse document frequency is the logarithm of the ratio of the total number of documents over the number of documents containing x.

We used a common  TF-IDF function:
```
#We compute the TF-IDF keys for the observations in the variables text and title
number_of_dimensions = 1000

#Vector representation of the text

tfidf_vectorizer = TfidfVectorizer(
    # Whether the feature should be made of word or character n-grams
    analyzer='word',
    # Unigrams: we consider one word by one word
    ngram_range=(1, 1),
    # Construct a vector the 1000 most used words
    max_features=number_of_dimensions,
    # Don't take the words that have a frequency higher than 100% 
    max_df=1.0,
    # Don't take the words that appear less than 10 times
    min_df=10)
```

We decided to put 1000 as maximum number of features, i.e. of words in a document. We are not using caracters n-grams, but only analyzing one word after one word. Finally, we want the most important words, so if they don't appear at least 10 times, we don't take them into account. 

* How does it work?

A SVM model aims to represent different classes in a hyperplane in multidimensional space. This hyperplane is generated using an iteration to minimize the errors. Then, the model divide the datasets into several classes to find the maximum marginal hyperplane. Then, the first step is to generate multiple hyperplanes with an iteration to segregate the classes the best as it can. The second step is to decide which hyperplane is the one that split the classes the best. There are several parameters that you can play on when trying to optimize the model, but here we won't optimize it as it works already well without optimization and without the features described in the literature. We only play on the tf-idf vectorization, and then on the frequency/ readability of the documents, not on the LIWC , punctuaction, CFG...
Here, our model only uses the tf idf vectorization of the text, i.e a quantitative variable, to segregate the two classes (fake or true).

We first charge our model from the sklearn libraries:
```lsvc = svm.LinearSVC(max_iter=300)```

Then, we fit our model on the train set:
```lsvc.fit(X_train,y_train)```

Finally, we predict the classes of the test set, using the features of the test set:
```pred_lsvc = lsvc.predict(X_test)```

To end, we compare these predicted values with the actual ones and we compute several metrics:
```
lsvc_accuracy = metrics.accuracy_score(y_test,pred_lsvc)
print("Training accuracy Score   : ", lsvc.score(X_train,y_train))
print("Testing accuracy Score : ", lsvc_accuracy)
print(metrics.classification_report(pred_lsvc,y_test))
```

* The results we got

The training accuracy score is 99,6%, while the test accuracy score is the 99,2%. We then don't really suspect overfitting as both accuracy scores are very close, and very good.  Looking at other metrics, we got the same results for the recall and f1 score, and for both classes.

```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      4721
           1       0.99      0.99      0.99      4259

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980 
```

The results are already well, almost perfect. We think it is due to the chosen dataset, that might be very easy to classify, as we saw on the literature that the classification and the accuracy scores vary a lot from a dataset to another.

### RNN

* What is an RNN? 

Deep neural networks usually assume independence between inputs and outputs. However, it isn't the case for RNNs. Indeed, RNNs have an internal memory thanks to a hidden state feature. It means that information are taken from previous inputs to influence the next input and output within each sequence. RNN are therefore known to understand better sequences and their context. 

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
Among the next layers are Dropout layers. There are used as a regularization technique. The principle in to shut some neurons down so they are less sensitive to the activation of another neuron.      
Each neuron in the dense layer receives input from all neurons of the previous layer and an activation function, here relu, can be associated to it.

Once all the previous steps where done we could configure our model with a loss, a metric and an optimizer. 
```
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
```

The binary crossentropy loss was chosen since the news can be only of two different types. As an optimizer was chosen ADAM (Adaptive Moment Estimation) which combines momentum and RMSprop. For the metrics, the accuracy is used. 

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

* What is a LSTM?

An LSTM model is a particular advanced form of RNN, that has been previously described. The main difference between the LSTM and the other RNN is that the hidden layer of LSTM is a gated unit or gated cell, i.e. four layers that interact with one to another to produce the output. LSTM has three logistic sigmoid gates and one tanh layer. The output is either 0 or 1. Those gates allows to control the memorizing process.

* How does it work?
 
 Here is the architecture of a LSTM.
 ![alt text](https://miro.medium.com/max/1400/1*Niu_c_FhGtLuHjrStkB_4Q.png)
 
 with:
 - X_t is the current input
 - x is the scaling of information
 - Where there is a +, it is where the information is added
 - The sigma is the sigmoid layer (it is an activation function). Here it is chosen because as it outputs 0 or 1, it can forget or remember information.
 - tanh is another layer that is also another activation function. Here it is chosen to overcome the increasing gradient problem (the second derivative of tanh doesn't converge to fast to 0).
 - h(t-1) is the output of the last LSTM unit, here consider as a new input
 - c(t) is the new updated memory
 - h(t) is the current output
 
 The particularity of the LSTM is its faculty to forget the unnecessary information with the sigmoid layer. 
 
* Steps

The steps are as follows:
- First, the sigmoid layer allows to forget unnecessary information from the previous unit, taking the input X(t) and h(t-1) and deciding which parts from the old output should be removed conditionnally to the new input.
- Then, it looks at the new imput and decide to store the information or not. Here, the sigmoid layer layer is the one that decide which part and how much of the new information the algorithm memorize. Tht tanh layer creates a vector of all the possible values. Then, both outputs from the two layers are multipled. 
- Finally, it has to decide the output. A sigmoid layer decide which parts and how much of the cell state the algorithm is going to output. It is again multiplied by the output of a tanh function that displays all the possible values. 

For our LSTM model, we fixed the maximum number of words taken into account by the model vocabulary as 10000, the maximum number of words per document as 100, and the dimension of the embedding layer in the network as 200.

Then, we decided to use the embedding layer from keras. It needs that the input data is encoded into integer, to have each word represented by a unique integer. The Tokenize function allows to do that, as follows:
```
# Tokenizer transforms sequences of word into sequences of index
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,.:;<=>?@[\\]^_`{|}~\t\n')

# We fit it on the X train set
tokenizer.fit_on_texts(X_train)

# We then vectorize the X train set
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post', truncating='post')

# We then vectorize the X test set
X_test = tokenizer.texts_to_sequences(X_test) 
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post', truncating='post')

# We set out-of index vocabulary to 0
X_train[X_train >= VOCAB_SIZE] = 0
X_test[X_test >= VOCAB_SIZE] = 0 
```

Below, we added some lines because we need to make all sequences in a batch to fit a given standard length. Then we padded (add some 0 to the end of a sentence) or truncated (remove the last words of a sentence) our sequences as follows:
- If the number of words of the document is lower than the maximum length defined above, then we do a padding.
- If the number of words of the review is higher than the maximum length defined above, then we do a truncating.

We then build ou LSTM model as follows:
```
def build_lstm_model(nb_class, voc_size, max_length, embedding_dim):

    inp = Input(shape=(max_length, ))

    x = Embedding(input_dim=voc_size,
                  output_dim=embedding_dim,
                  input_length=max_length,
                  trainable=True)(inp)

    x = Bidirectional(LSTM(128))(x)
    out = Dense(nb_class, activation='sigmoid', name='output')(x)

    model = Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(lr=0.005)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model

lstm_model = build_lstm_model(
    MODEL_OUPUT_DIM, 
    VOCAB_SIZE,
    MAX_LENGTH,
    EMBEDDING_DIM)
```
We clearly can see the sigmoid layer given as activation function (the tanh being in the Bidirectional LSTM function). We then fit our model on the train set using several new parameters:
- epochs= 2, i.e the number of time we're going to run our network
- batch_size: the number of documents that will be passed throught the network at one time
- verbose= 1: It shows all the steps done as output
- validation_split=0.1: the size of the validation set
 
 ```
history = lstm_model.fit(
    X_train,
    y_train.values,
    epochs=2,
    batch_size=1024,
    verbose=1,
    validation_split=0.1
    )
```

Then, thanks to the two functions below that we defined, we can plot some metrics to measure our classification.

```
def plot_history(hist):
  plt.plot(hist.history['loss'], label='train')
  plt.plot(hist.history['val_loss'], label='val')
  plt.legend()
  plt.ylim((0,1))
  plt.title('Loss evolution')
  plt.show()
  plt.plot(hist.history['binary_accuracy'], label='train')
  plt.plot(hist.history['val_binary_accuracy'], label='val')
  plt.legend()
  plt.ylim((0,1))
  plt.title('Accuracy evolution')
  plt.show()
  
  def model_evaluation(NN_model, x_test, y_test): 
  y_pred_proba = NN_model.predict(x_test, verbose=1)
  y_pred = [1 if i >= 0.5 else 0 for i in y_pred_proba]
  print('accuracy {}'.format(round(accuracy_score(y_test, y_pred), 4)))
  
  plot_history(history)
model_evaluation(NN_model=lstm_model, x_test=X_test, y_test=y_test)
```

* Result

The results of the fitting are as follows. It lets us imagine that the network classify very well on the train set and on a small validation set, as the accuracy are higher than 99%. Two epochs are clearly enough. 
```
Epoch 1/2
32/32 [==============================] - 169s 5s/step - loss: 0.1948 - binary_accuracy: 0.9233 - val_loss: 0.0286 - val_binary_accuracy: 0.9897
Epoch 2/2
32/32 [==============================] - 153s 5s/step - loss: 0.0089 - binary_accuracy: 0.9977 - val_loss: 0.0094 - val_binary_accuracy: 0.9969

```
On the test set, we get an accuracy score of 99,8% , which is almost perfect. Here again, it is probably due to the choice of dataset.


### BERT

* What is an BERT?

The specificity of BERT models is they use the transformer encoder architecture to process each input text token in the full context of all tokens before and after. Several BERT models exists. In this project we used the bert base model, which consists of 12 layers of transformer encoder, 12 attention heads, 768 hidden size, and 110M parameters.

* How does it work? 

The transformer is the part of the model that enables BERT to outperform other models. The transformer processes a given word in relation to all other words in the sentence, rather than processing them one by one. It allows to fully understand the context of the word. 

BERT models are pre-trained on a large corpus of text (English Wikipedia 2,500M words and BooksCorpus 800M words) and then refined for specific tasks. BERT is bidirectional, it learns information from a sequence of words from the left and right. Furthermore, it is pre-trained on two tasks. The first one is Masked Language Model: the model is trained by hiding a word in a sentence and then trying to predict it based on the masked word's context. The second one is Next Sentence Prediction: this time the model is training by looking at two sentences and predicts if they have a sequential connection or just a random one. 
An attention process is used, it means that each output element is connected to every input element, and the weights between them are dynamically calculated based upon their connection. 

It is necessary to train the BERT model on a new task to specialize the model, this is called finetuning. Indeed, BERT is pre-trained using only an unlabeled text corpus, thus additional layers of neurons need to be added to transform the model into a classifier example.

* Issue with BERT 

The issue we got with BERT is that is was running for several hours. Not having computers powerfull enough and still wanting a result we decided to reduce the dataset to 5000 news. Obviously, the results gotten aren't representative of BERT's performance and the results would be much better if using the whole dataset. 
We could have used another BERT model like the DistilBERT for example, however, it is mostly certain we would have had the same issue. 

* Steps 

BERT is using token wods, thus before anything else, we had to define the tokenizer and maximum length.

```
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
max_len   = tokenizer.max_model_input_sizes['bert-base-uncased']
```

In order to generate new data, we create a dataset class to customize the data. 

```
class Dataset(torch.utils.data.Dataset): 
    
    def __init__(self,df): 
        '''
        Get labels and tokenization of the text
        '''
        self.labels = [labels[label] for label in df["class"]] 
        self.texts = [tokenizer(text, padding='max_length', max_length=max_len, 
                                truncation=True,return_tensors="pt") for text in df["text"]] 
    
    def classes(self):
        return self.labels
    
    def __len__(self): 
        return len(self.labels)
    
    def get_batch_labels(self,indx): 
        '''
        Batch of labels
        '''
        return np.array(self.labels[indx])

    def get_batch_texts(self,indx): 
        '''
        Batch of texts
        '''
        return self.texts[indx]

    def __getitem__(self,indx): 
        '''
        Item with the labels and texts
        '''
        batch_y = self.get_batch_labels(indx)
        batch_texts = self.get_batch_texts(indx)
        
        return batch_texts, batch_y
```

We then built the model. As explained we used a pre trained model from BERT. A dropout layer is added, as well as a linear one that applies a linear transformation to the input data. There are two variables here, one that contains the embedding vectors of all of the tokens in a sequence and the other one the classification embedding vectors. At the end, we get a vector of size 2 corresponding to the two possible labels. 

```
class BertClassifier(torch.nn.Module): 
    
    def __init__(self): 
        super(BertClassifier,self).__init__()
        
        self.bert=BertModel.from_pretrained("bert-base-cased")
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768,6) 
        
    def forward(self,input_id,mask): 
        
        _,pooler_output = self.bert(input_ids= input_id,attention_mask = mask,return_dict = False)
        dropout_output = self.dropout(pooler_output)
        linear_output  = self.linear(dropout_output)
        
        return linear_output
```

In the training we define a loss function and an optimizer as done for the previous models. A binary crosstropy could have been used here since we have only two possible classes. The function enables to train the model, including the pre-processing module, the BERT encoder, the data and the classifier.

```
def train(model, train_data, valid_data, learning_rate, epochs=1):
    
    # Create custom data
    train, valid = Dataset(train_data), Dataset(valid_data)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid, batch_size=1)
    
    # Processor 
    device = torch.device("cpu")
    
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
   
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask) # Prediction
                
                # Get loss
                batch_loss = criterion(output, train_label) 
                total_loss_train += batch_loss.item()
                
                # Get accuracry
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                
                # Update the model
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
            # Same procedure on the validation data
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in valid_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(f'Epochs: {epoch_num + 1} \n\
Train loss: {total_loss_train / len(train_data):6f} \n\
Train accuracy: {total_acc_train / len(train_data):6f} \n\
Validation loss: {total_loss_val / len(valid_data):6f} \n\
Validation accuracy: {total_acc_val / len(valid_data):6f}')

train(model = BertClassifier(), train_data = df_train, valid_data = df_valid, learning_rate = 1e-6, epochs = 1)
```

Finally, we evaluated the model on the test data. To evaluate the model we could have plotted the loss and accuracy, but having several epochs would have taken too long. 

```
def evaluate(model, test_data):
    
    # Create custom data
    test = Dataset(test_data)
    
    # Create dataloaders
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    # Processor 
    device = torch.device("cpu")

    total_acc_test = 0
    
    with torch.no_grad():
        
        # Prediction and accurary computation
        for test_input, test_label in tqdm(test_dataloader):

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data):6f}')

evaluate(model = BertClassifier(), test_data = df_test)
```

* Results

We got good results while validating our model, both the accuracy of the train and validation data where above 92%. 

Train loss: 0.288873    
Train accuracy: 0.925750    
Validation loss: 0.029467    
Validation accuracy: 0.996000         

Nonetheless, we got a test accuracy of 48%. This can be explained by the fact that we reduced the datasets. We believe that if we had more data we would have gotten very good results. 

BERT being very slow, we didn't focus on this model too much. Thus, it could be largely improved. 

Using BERT was a first attempt to understand a little more how this kind of model works. 

## Conclusion

All of our algorithms got pretty good results. The accuracy scores were all high, except for the BERT model but because as we explained, we tried on a subsample. Those very good results, compared to the ones in the literature, let us think that the choice of dataset might be relevant in the text classification problem, and maybe we would not get as good results with other data. 

The models that have the best accuracy scores on the test set is the LSTM. It is compliant with what was seen in the literature and with our courses on Deep Learning. It is secondly followed by the LSVM and then by the RNN. It shows us that sometimes a non-neural network model can work better than a neural network.

The BERT model, even if here the result are not really good on the test set, is very promising. Indeed, the validation and training accuracy scores are very good, just not replicable on the test set, due to the fact that we tried it on subsample. To go further, it would be nice to try it on the whole dataset or on another one. In our case, it would have taken 4 hours to run, and our computers were not powerful enough to do it as the same time as our daily tasks. 

Another idea to go further would have, as they do it in the literature, to test our algorithms on other datasets to see their robustness to other features. We decided to no do it by lack of time. 

Finally, another idea would be to add other features on the LSVC model, such as the title, the subject, but also the punctuation, the readability, the n-grams.... However, in our case, without all of it the results were already very good. 

We enjoyed working on this project as it allowed us to apply algorithms that we already knew to text classfication, but alo to discover new ones that are specific to text classification.

