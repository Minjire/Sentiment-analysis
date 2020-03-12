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

'''
Model commented as it had been saved. To train model uncomment and comment the remaining part of code

# define model
model = keras.Sequential()

# word embedding layer to attempt to group words with almost similar meaning/determine the meaning of each word....
# in the sentence by mapping each word to a position in vector space in this case of dimension 16
model.add(keras.layers.Embedding(100000, output_dim=16, input_length=250))

# scales down our data's dimension to make it easier computationally for our model in the later layers
model.add(keras.layers.GlobalAveragePooling1D())

# contains 16 neurons with a relu activation function designed to find patterns between different words present in the review
model.add(keras.layers.Dense(16, activation="relu"))

# sigmoid function to get a value between a 0 and a 1 representing the likelihood of the review being positive/negative
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

# Compiling the model is just picking the optimizer, loss function and metrics to keep track of
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# split data to training and validation
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# train model; batch_size--no. of reviews loaded each cycle
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print("\n")
print(results)

model.save("model1.h5")  # name it whatever you want but end with .h5

#load model
model = keras.models.load_model('model1.h5')
# convert model to tensorflow lite file
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("model1.tflite", "wb").write(tflite_model)


'''

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


messages = []
count = 0

with open("whatsapp_texts.txt", encoding="utf8") as fh:
    for line in fh:
        # remove leading and trailing characters
        line = line.strip()
        # skip blank lines
        if line:
            time, description = line.strip().split('-', 1)
            name, message = description.strip().split(':', 1)
            messages.append({"time": time, "name": name, "message": message.strip()})
            count += 1
        else:
            continue

for i in messages:
    text = i['message']
    nline = text.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace \
        ("\"", "").replace("/", "").replace("-", "").replace("?", "").replace("!", "").replace(">", "").replace \
        ("<", "").replace("'", "").strip().split(" ")
    encode = review_encode(nline)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)  # make the data 250 words long
    predict = model.predict(encode)
    print("Sender: " + i['name'] + "\t\tText: " + text + "\t\tTimestamp: " + i['time'])
    # print(encode)
    print(predict[0])
