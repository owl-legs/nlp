import pickle
import numpy as np
from tensorflow.keras.models import load_model

class TestInput():
    def __init__(self):
        self.tokenizer = pickle.load(open('embeddings/maps/oneHotEmbeddingMap', "rb"))
        self.tokenKeyMap = {v:k for k,v in self.tokenizer.items()}
        self.model = load_model("metamorphasis.h5")

    def __predict_next_token__(self, lastToken):
        input = [self.tokenizer.get(lastToken, 100)]
        index = np.argmax(self.model.predict(input))
        return self.tokenKeyMap[index]
    def __test__(self):
        while True:
            text = input("Enter text: \n")
            if text.lower() == "quit":
                break
            else:
                lastToken = ((text.lower()).split(" "))[-1]
                nextWord = self.__predict_next_token__(lastToken)
                print(nextWord)
    def start(self):
        self.__test__()

def main():
    TestInput().start()

main()
