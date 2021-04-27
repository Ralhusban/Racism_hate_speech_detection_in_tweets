# Racism_hate_speech_detection_in_tweets

Problem statement

Social media is having an increasing influence on the world, and in the past years we’ve seen violence instigated by social media. Twitter is a well known platform that helped people spread news around the world in an unprecedented manner. However, at the same pace, hate, violent, and racism was equally propagated by some users of Twitter. Companies such as Twitter and Facebook strive to improve their hate speech / racism detection algorithms to flag and remove any tweets / posts containing such language. Therefore, an efficient design of such model would be beneficial overall to counter the spread of hate speech and racism on social media.

2. Dataset


I used a Twitter tweets dataset provided by Analytics Vandhya as a competition. The dataset contains 32’000 labelled tweets which will be used to train and test our model. The highest achieved score (F-1) in the competition is ~87%. The Data set is highly unbalanced – containing only 7% training data with hate speech / racism while the rest is regular tweets.


4. Technologies  used:

4.1: Packages

- NLTK 
- Keras
- Hugging face (Destilled-BERT)
- Scikit-learn
- Numpy
- Pandas
- Matplotlib


4.2 Models / Architecture


While there have been many attempts at this problem using tokenization techniques and using Bag of Words and Term Frequency – Inverse Term Frequency (TF-IDF) embedding methods, I used a different approach for embedding the data. A pretrained Distilled BERT model embedding as a first step, and separately use TF-IDF encoding and combine features of both methods. Prior to concatination of both feature vectors, TF-IDF features were converted to dense using the built-in feature in the Sklearn’s implementation of TF-IDF ‘.todense()’ for efficiency reasons.  Afterwards, the resulting data will be passed into a Densly Connected Deep Neural Network for classification

5. Preprocessing

The input tweets were preprocessed with the following methods:

-	Removal of 1 charachter words
- 	Removal of digits
- 	Lemmatization (Proved to be more efficient than stemming in preserving contextual value)
-	Removal of stop words
-	Removal of special charachters such as @, #, ! , etc.

Note that despite many sources pointing the deep learning architecture doesn’t require pre-processing of text data, I found the contrary. Despite that DNNs are inherently designed to extract features, text pre-processing still enhances the model’s performance. This could be related to BERT embeddings being more efficient when the input text is cleaned up of words and characters that have little or no contextual value.

6. Architecture Design, fine-tuning, and discussion





Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 128)               4214272   
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 129       
=================================================================
Total params: 4,247,425
Trainable params: 4,247,425
Non-trainable params: 0


The model structure is rather simple (cf drawing 3): It contains two hidden layers with RELU activation functions. Dropout of 40% are added after each layer with an L2 regularization of 1% at each of the hidden layers. The final layer is sigmoid activated for our binary classification problem.

I have experimented with L2 and drop out ratios and found these to provide the best accuracy and f1-score. Moreover, another unconventional fine-tuning method was to expirement with testing the sigmoid activtication threshold vis-a-vis prediction with 1 or 0. This was done thru a simple for loop from 1 to 100 that tested the f-1 score at each run and the optimal ‘threshold’ was actually found at 0.33 instead of the standard 0.50 cut-off. This increased F1-Score by about 2.5% (c.f. Fig.10)




