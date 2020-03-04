import tensorflow as tf
from tensorflow import keras
import numpy as np

# get data from keras
data = keras.datasets.imdb

# split data into training and testing
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=100000)

# A dictionary mapping words to an integer index
word_index = data.get_word_index()

# break dictionary/tuple to key and value
word_index = {k: (v + 3) for k, v in word_index.items()}

# 3 values added
word_index["<PAD>"] = 0  # pad tag to make review same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# reverse that dictionary so we can use integers as keys that map to each word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


#  this function will return the decoded (human readable) reviews
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# define data inputs to be of same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# load model
model = keras.models.load_model("model1.h5")


def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


text = input("Enter a message in English or 'y' to exit: ")
while text != 'y':
    nline = text.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace \
        ("\"", "").replace("/", "").replace("-", "").replace("?", "").replace("!", "").replace(">", "").replace \
        ("<", "").replace("'", "").strip().split(" ")
    encode = review_encode(nline)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)  # make the data 250 words long
    predict = model.predict(encode)
    print(text)
    print(encode)
    print(predict[0])
    text = input("Enter a message in English or 'y' to exit: ")
