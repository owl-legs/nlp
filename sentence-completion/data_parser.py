import config
import os
import re

def files(directory=config.TRAINING_DATA_PATH):
    return list(os.listdir(directory))

def extract_text(lines):
    lines_no_whitespace = re.findall("\S.*\S", lines)
    return '' if not lines_no_whitespace else lines_no_whitespace[0]

def clean(text):
    words = re.findall("[a-zA-Z'-]+", text)
    return ' '.join(words) if words else ''

def load_data(directory=config.TRAINING_DATA_PATH):
    trainingData = []
    book = ''
    #with open('data/sentences.txt', 'w') as output:
    for file in files():
        with open(f'''{directory}/{file}''', 'r') as novel:
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
            book += (clean(text))
            text = [clean(s).split(' ') for s in text.split(".")]
            trainingData += [text]
    print(trainingData[0])

load_data()