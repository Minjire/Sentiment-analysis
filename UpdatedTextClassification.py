import pickle
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

# %% load and combine datasets
df1 = pd.read_csv('imdbdata.csv', index_col=0)
df2 = pd.read_csv('refinedsentimenttestdata.csv', header=None)
df3 = pd.read_csv('refinedsentimenttraindata.csv', header=None)

df2.columns = ['Review', 'Label']
df3.columns = ['Review', 'Label']

df = pd.concat([df1, df2, df3])
print(df.head())

# %% load dictionary containing word index
pkl_file = open('index_dictionary.pkl', 'rb')
word_index = pickle.load(pkl_file)
pkl_file.close()

# reverse that dictionary so we can use integers as keys that map to each word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# save dictionary to txt file
with open("index.txt", "w", encoding='utf-8') as myfile:
    for key in sorted(reverse_word_index):
        myfile.write(reverse_word_index[key] + " " + str(key) + "\n")


#  this function will return the decoded (human readable) reviews
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# %% encode review column
predict = "Label"
encoded_x = []
x = np.array(df.drop([predict], 1))
y = np.array(df[predict])


def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


for i in range(len(x)):
    row = str(x[i]).replace("[", "").replace("]", "").replace("\"", "").split()
    row.pop(0)  # remove <START> tag
    encoded_x.append(review_encode(row))
    if (i % 1000) == 0:
        print(i)
print(i)

# %%
print(len(x))
print(len(encoded_x))

# %% split data to train and test
train_data, test_data, train_labels, test_labels = train_test_split(encoded_x, y, test_size=0.2)

# define data inputs to be of same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# %% model development
# define model
model = keras.Sequential()

# word embedding layer to attempt to group words with almost similar meaning/determine the meaning of each word....
# in the sentence by mapping each word to a position in vector space in this case of dimension 16
model.add(keras.layers.Embedding(440000, output_dim=16, input_length=250))

# scales down our data's dimension to make it easier computationally for our model in the later layers
model.add(keras.layers.GlobalAveragePooling1D())

# contains 16 neurons with a relu activation function designed to find patterns between different words present in
# the review
model.add(keras.layers.Dense(16, activation="relu"))

# sigmoid function to get a value between a 0 and a 1 representing the likelihood of the review being positive/negative
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

# Compiling the model is just picking the optimizer, loss function and metrics to keep track of
model.compile(optimizer="", loss="binary_crossentropy", metrics=["accuracy"])

# split data to training and validation
x_val = train_data[:100000]
x_train = train_data[100000:]

y_val = train_labels[:100000]
y_train = train_labels[100000:]

# train model; batch_size--no. of reviews loaded each cycle
# validation_data=(x_val, y_val)
fitModel = model.fit(train_data, train_labels, epochs=5, batch_size=256, validation_split=0.2, verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

# %%
model.save("refinedmodel.h5")  # name it whatever you want but end with .h5

# %% convert model to tflite
# load model
model = keras.models.load_model('refinedmodel1.h5')
# convert model to tensorflow lite file
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("refinedmodel1.tflite", "wb").write(tflite_model)
