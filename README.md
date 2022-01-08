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















## Conclusion


