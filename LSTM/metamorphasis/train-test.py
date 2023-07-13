import pickle
import config
import numpy as np

tokenizedSentences = pickle.load(open(config.TOKENIZED_SENTENCE_OUTPUT, 'rb'))
allText = np.array(list(map(lambda x: x[1:-1], sentences))).flatten()

sequences = []
for i in range(1, len(allText)-1):
    sequences.append(allText[i-1:i+1])

X = list(map(lambda x: x[0], sequences))
y = list(map(lambda x: x[1], sequences))

