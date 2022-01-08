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

Text classification is a popular task in Natural Language Processing (NLP). The purpose is to classify text based on pre-defined classes, usually sentiments or topics. Text can contain a lot of information, however extracting this infromation can take time, especially due to its unstructured nature. With nowadays's deep learning models classify text is getting easier. Fake news are spreading false information in order to influence readers' beliefs which usually damages a person's, a company's... reputation. Thus, they are a real issue today, especially since a vast volume of text data is generated everyday in the form of social media, websites, news... etc. 

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

The first paper details the challenges, the tasks, and the NLP solutions to answers to the challenge of fake news detection. They first transformed the problem into a binary classification problem (fake - real), but then added other classes for not completely real and not completely fake news. According to them, the following model can be used for text classification: 
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
- Punctuation: twelve types of punctuaction derived from the Linguistic Inquiry and Word COunt software (LIWC) are used.
- LIWC: It is a lexicon that allows to extract the proportions of words into several ategories. It can represents psycholinguisitic processes, summary categories, part-of-speech categories...
- Readability: Those are features to indicate text understandability (number of characters, complex words...)
- Ngrams: It aims to extract unigrams and bigrams derived from the bag of words representatio of each news articles. These features are then tf-idf vectorized. 
- CFG (Context free grammars): It is a tree that is composed of all the lexicalized production rules combiened with their ancient nodes. These features are also tf-idf vectorized.
- All features

For the first dataset, the best results is when only using the readability, and then the acuracy score is 78%. When using all the features, we get an accuracy score of 74%. The opposite is seen on the second dataset: the accuracy score when only using the readability is the worst (62%). The best accuracy scores is when using all features (76%) and when using the LIWC lexicon (74%). This paper clearly shows the importance of the choice of dataset for the classification problem. 

## Data

The data used come from kaggle: *Fake and real news dataset* (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). It contains a first dataset with 23481 fake news and a second one with 21417 true news. Each of the datasets has 4 columns: the title, the text, the subject and the date. In our models we will mainly use the text and sometimes the title. To be able to use the data, it had to be cleaned. 

## Data cleaning 

We had two datasets but needed just one, thus the first thing we did was to merge the fake and true news datasets and specified each time the class (true or fake) of the news. Once this step was done, we could clean the title and text using the Natural Language Toolkit.  

We created several functions:
- Convert_text_to_lowercase: It allows to put all the charactes in lowercase.
- Remove punctuation: It allows to remove some useless punctuation.
- Tokenize_sentence: It allows to tokenize the sentence, i.e., to split the word one
by one as vectors of letters.
- Remove_stop_words: It allows to remove some basic English words such as "the", "a", "an"...
- Lemm: It keps only the roots of words. 
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

In order to generate the news data, we create a dataset class to custom the data. 

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


