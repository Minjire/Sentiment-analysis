from tensorflow import keras
import pandas as pd
import operator
import pickle

# %% loading data
imdb = keras.datasets.imdb

# splitting our dataset into training and testing data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

# converting integer encoded words to string
# a dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in _word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2  # unknown
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# will return the decoded (human readable) reviews
def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])


# dictionary to store the reviews
imdb_data = {'Review': [], 'Label': []}
for i in range(len(train_data)):
    imdb_data['Review'].append(decode_review(train_data[i - 1]))
    imdb_data['Label'].append(train_labels[i - 1])

for i in range(len(test_data)):
    imdb_data['Review'].append(decode_review(test_data[i - 1]))
    imdb_data['Label'].append(test_labels[i - 1])

# store reviews in a data frame
df = pd.DataFrame(imdb_data)
df.to_csv('imdbdata.csv')
print(df)

word_index_dict = {'Word': [], 'Index': []}
for key, value in word_index.items():
    word_index_dict['Word'].append(key)
    word_index_dict['Index'].append(value)

print(max(word_index_dict['Index']))
if 'hate' in word_index_dict['Word']:
    hate_index = word_index_dict['Index'][word_index_dict['Word'].index('hate')]
    print(hate_index)

print(decode_review([hate_index]))

# %% dataset to be combined

df_test = pd.read_csv('refinedsentimenttestdata.csv', header=None)
df_train = pd.read_csv('refinedsentimenttraindata.csv', header=None)
# insert column labels
df_test.columns = ['Review', 'Label']
df_train.columns = ['Review', 'Label']

print(df_test.head())
print(df_train.head())

# %% check if word is in dictionary and store in array
missing_words = {}


def missingwords(dframe):
    for index in dframe.index:
        row_list = dframe.loc[index, "Review"]
        for word in row_list.split():
            if word in word_index:
                None
            else:
                if word in missing_words:
                    missing_words[word] += 1
                else:
                    missing_words[word] = 1
    if index % 1000 == 0: print(index)


missingwords(df_test)
missingwords(df_train)

# %% sort dictionary descending order by frequency of word occurrence
sorted_missing_words = dict(sorted(missing_words.items(), key=operator.itemgetter(1)))
# remove values with occurrence of 1, may cause noise
sorted_missing_words = {key: value for key, value in sorted_missing_words.items() if value != 1}

# create new word index
new_word_index = {}
new_word_index = {**new_word_index, **word_index}
i = len(new_word_index)

for word in sorted_missing_words:
    i += 1
    new_word_index.update({word: i})
    if i % 1000 == 0: print(i)

# save dictionary of indexes
output = open('index_dictionary.pkl', 'wb')
pickle.dump(new_word_index, output)
output.close()


