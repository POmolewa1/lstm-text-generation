import os

from sympy.codegen.ast import continue_

from local_code.joke_generation_code.tokenizer import *

class Simple_Setting():
    model = None

    def __init__(self,datapath,mod):
        self.model = mod
        self.set_dataset(datapath)
        #print(self.model.testing_set)


    def set_dataset(self, datapath):
        if datapath == "data/joke_generation_data/text_generation/data.csv":
            self.model.dataset = create_joke_data(datapath)
        else:
            self.model.dataset = create_dataset(datapath + "/train")
            self.model.testing_set = create_dataset(datapath + "/test")

    def train(self):
        self.model.start()


    def test_model(self):
        self.model.testing_process()



