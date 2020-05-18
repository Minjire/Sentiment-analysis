import pandas as pd
import numpy as np
df = pd.read_excel('sentimenttestdata.xlsx', header=None)
df.head()

# name columns and add column
df.columns = ['Review']
df['Label'] = ""
df.head()

df['Label'] = np.where(df.Review.str.contains("__label__2"), 1, 0)
df.head()

tag = '<START>'

# remove label in string and save dataframe to csv
for index in df.index:
    row_list = df.loc[index, "Review"].replace(",", "").split(" ", 1)
    row_list[1] = row_list[1].replace(";", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "") \
        .replace("\"", "").replace(" - ", "").replace("--", "").replace("- ", "").replace("?", "").replace("!", "")\
        .replace("[", "").replace("]", "").replace("*", "").replace("_", "").replace(" /", "").strip().lower()
    row_data = tag + " " + ('{},{}'.format(row_list[1], df.loc[index, "Label"]))
    with open("refinedsentimenttestdata.csv", "a", encoding="utf-8") as file:
        file.write(row_data + '\n')
    if (index % 1000) == 0:
        print(index)

print(index)

