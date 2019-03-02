
# coding: utf-8

# In[ ]:


import nltk
import tensorflow as tf
import keras
from gensim.models import Word2Vec
import multiprocessing
import os
from keras.initializers import Constant
import matplotlib.pyplot as plt
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical

# In[ ]:


def load_document(filename):
    file = open(filename, 'r', encoding='utf-8')
    text = file.read()
    file.close()
    return text


# In[ ]:


doc = load_document('republic_clean.txt')
print(doc[:200])


# In[ ]:


import string
 
# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


# In[ ]:


tokens = clean_doc(doc)
print(tokens[:200])
print("Total Tokens: %d" % len(tokens))
print("Unique Tokens: %d" % len(set(tokens)))


# In[ ]:


# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))


# In[ ]:


def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# In[ ]:


out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)


# In[ ]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
 
# load
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')


# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)


# In[ ]:


# vocabulary size
vocab_size = len(tokenizer.word_index) + 1


# In[ ]:


# separate into input and output
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


# In[ ]:


# define model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, CuDNNLSTM, Dropout
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=seq_length))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(128)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# In[ ]:


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history=model.fit(X, y, batch_size=256, epochs=50, validation_split=0.2)

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("acc.png")

plt.figure( )
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("loss.png")



