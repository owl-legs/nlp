import pickle
import config
import numpy as np
from tensorflow.keras.models import load_model

class TestInput():
    def __init__(self):
        self.tokenizer = pickle.load(open(config.TOKEN_DIC_OUTPUT, "rb"))
        self.tokenKeyMap = {v:k for k,v in self.tokenizer.items()}
        self.model = load_model("metamorphasis.h5")

    def __predict_next_token__(self, lastToken):
        input = np.array(self.tokenizer[lastToken])
        preds = self.model.predict_classes(input)
        index = np.argmax(preds)
        return self.tokenKeyMap[index]
    def __test__(self):
        while True:
            text = input("Enter text: \n")
            if text.lower() == "quit":
                break
            else:
                lastToken = ((text.lower()).split(" "))[-1]
                self.__predict_next_token__(lastToken)
    def start(self):
        self.__test__()

def main():
    TestInput().start()

main()
