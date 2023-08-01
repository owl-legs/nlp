import config
import pickle
import codecs
import os
import re
import pandas as pd

def files(directory):
    return list(os.listdir(directory))

def extract_text(lines):
    lines_no_whitespace = re.findall("\S.*\S", lines)
    return '' if not lines_no_whitespace else lines_no_whitespace[0]

def clean(text):
    words = re.findall("[a-zA-Z'-]+", text)
    words = ["".join(word.split("'")) for word in words]
    return ' '.join(words) if words else ''

def load_train_data(directory=config.TRAINING_DATA_PATH):
    trainingData = []
    with codecs.open('model/all_words.txt', "w", encoding='utf-8', errors='ignore') as out:
        for file in files(directory):
            with open(f'''{directory}/{file}''', 'r') as novel:
                print(f'''parsing {directory}/{file}''')
                text = ''
                parseFlag = False
                for line in novel.readlines():
                    line = line.replace("\n", " ")
                    if line and parseFlag:
                        if line.find('----------') > -1:
                            break
                        text += (extract_text(line.lower()) + ' ')
                    if not parseFlag and line.find('Arthur Conan Doyle') > -1:
                        parseFlag = True
                out.write(clean(text) + ' ')
                trainingData += [clean(s).split(' ') for s in text.split(".")]
        pickle.dump(trainingData, open('data/trainingData.txt', "wb"), True)

def load_data():
    load_train_data()

load_data()