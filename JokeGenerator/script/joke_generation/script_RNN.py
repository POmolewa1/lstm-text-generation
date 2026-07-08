import os
import sys
from asyncio.windows_events import NULL

import numpy as np
import torch

from local_code.joke_generation_code.Simple_Setting import Simple_Setting
from local_code.joke_generation_code.RNN_Gerneration import RNN_Generation


# get the directory of the current script
file_path = os.path.dirname(__file__)
# Change the working directory two files out
working_directory = os.path.join(file_path,'../../')
os.chdir(working_directory)
# Change the system path two files out
sys.path.append(str(working_directory))
sys.path.insert(0, str(working_directory))
#print(os.getcwd())


if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------
    joke_data = "data/joke_generation_data/text_generation/data.csv"


    load = int(input("Would you like to load a previously saved model? | (0) for no / (1) for yes: "))

    model = RNN_Generation()
    data_file = joke_data

    setting = Simple_Setting(data_file, model)
    if load:
        if setting.model.load_prev_model():
            setting.test_model()
        else:
            sys.exit(-1)
    else:
        setting.train()
