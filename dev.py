import os
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))


df = pd.DataFrame(columns=['review', 'class'])

path = "/Users/judithrosell/Desktop/MOVIE REVIEWS 2/dev"

for directory in os.listdir(path):
    if os.path.isdir(path + "/" + directory):
        for file in os.listdir(path + "/" + directory):
            with open(path + "/" + directory + "/" + file, encoding='utf-8') as f:
                if not file==".DS_Store":
                    classe = str(directory)
                    review = f.read()
                    words = re.sub(r"\W+", " ", review)
                    tokens = nltk.word_tokenize(words)
                    filtered_words = []
                    for token in tokens:
                        if token not in stopWords:
                            token = wnl.lemmatize(token)
                            filtered_words.append(token)
                    lemas_output = TreebankWordDetokenizer().detokenize(filtered_words)
                    new_row ={"review":lemas_output, "class":classe}
                    current_df = pd.DataFrame({'review': lemas_output, 'class': [classe]})
                    df = df.append(new_row, ignore_index=True)


df.to_csv("dev.csv")
