import torch
import numpy as np
import torch.nn as nn
import random

from sklearn.externals.array_api_compat import to_device
from sympy.codegen.ast import none
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_folder = Path("result/joke_generation_saves/outputs")

class RNN_Generation(nn.Module):
    dataset = None
    testing_set = None

    max_sentence_size = 25
    #max_sentence_size = 100
    #hidden_size = 128
    #hidden_size = 160
    hidden_size = 256

    # the number of unique words
    vocab_size = 20000
    # size of the embedding vector
    embedding_size = 50

    word_bank = []
    embedding_bank = []


    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.load_glove_embedding()
        self.embedding.weight.data.copy_(self.embedding_bank)

        self.embedding.weight.requires_grad = False
        #for i in range(self.vocab_size):
        #   print(self.embedding.weight[i])
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        #self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.to(device)

    def load_prev_model(self):
        path = Path("result/joke_generation_saves/model/rnn_gen.pth")
        if path.exists():
            self.load_state_dict(torch.load(path))
            return True
        else:
            print("No previous model found")
            return False

    def forward(self,x, hidden = None):
        x = self.embedding(x)

        # use this for hidden when using rnn or gru
        #_,hidden = self.rnn(x)

        output, hidden = self.rnn(x,hidden)
        output = self.fc(output)
        return output, hidden


    def training_process(self):
        print("Starting training process...")
        self.train()

        #max_epochs = 100
        max_epochs = 100
        learning_rate = 0.005

        # this avoids training our padding vector
        loss_function = nn.CrossEntropyLoss(ignore_index=0)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        text= self.make_tensor_data(self.dataset)

        batch_size = 100



        loss_history = []

        report_file = output_folder / "training_report.txt"

        with open(report_file, "w") as f:
            with open(report_file, "w") as f:
                f.write("Joke Generator Training Report\n")
                f.write("=" * 40 + "\n\n")
                f.write("Training Parameters\n")
                f.write("-" * 20 + "\n")
                f.write(f"Epochs: {max_epochs}\n")
                f.write(f"Learning Rate: {learning_rate}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"Hidden Size: {self.hidden_size}\n")
                f.write(f"Embedding Size: {self.embedding_size}\n")
                f.write(f"Vocabulary Size: {self.vocab_size}\n")
                f.write(f"Max Sentence Length: {self.max_sentence_size}\n")
                f.write(f"Optimizer: Adam\n")
                f.write(f"Loss Function: CrossEntropyLoss(ignore_index=0)\n")
                f.write(f"Device: {device}\n")
                f.write("\n")

        adjusted_dataset = TensorDataset(text)

        for epoch in range(max_epochs):
            dataloader = DataLoader(adjusted_dataset, batch_size=batch_size, shuffle=True)
            count = 0
            cumulative_loss = 0
            for i,batch in enumerate(dataloader):

                txt = batch[0].to(device)
                txt_input = txt[:, :-1]
                target = txt[:, 1:]

                prediction,_ = self.forward(txt_input)
                prediction = prediction.reshape(-1,self.vocab_size)
                target = target.reshape(-1)

                #print(torch.argmax(prediction,dim=1))

                optimizer.zero_grad()

                loss = loss_function(prediction, target)

                loss.backward()
                optimizer.step()

                cumulative_loss += loss.item()
                count += 1

            avg_loss = cumulative_loss / count

            loss_history.append(avg_loss)

            print(f"Epoch {epoch + 1}: {avg_loss:.4f}")

            with open(report_file, "a") as f:
                f.write(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}\n")

        plt.figure(figsize=(8, 5))
        plt.plot(loss_history)

        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Training Loss")
        plt.grid(True)

        plt.savefig(output_folder / "training_loss.png")
        plt.close()

        with open(report_file, "a") as f:
            f.write(f"\n")

        print("training finished saving to result/joke_generation_saves/model...")
        torch.save(self.state_dict(), "result/joke_generation_saves/model/rnn_gen.pth")


    def testing_process(self):
        print("Starting Testing process...")

        report_file = output_folder / "training_report.txt"

        self.eval()

        test_joke = ["why","did","the"]
        test_joke2 = ["can","february","march"]
        test_joke3 = []

        for x in range(3):
            test_embedding_index = random.randint(3,len(self.word_bank) - 1)
            word = self.match_number_to_word(test_embedding_index)
            test_joke3.append(word)

        test_joke_list = [test_joke, test_joke2, test_joke3]

        max_gen = 18
        with torch.no_grad():
            for joke_instance in test_joke_list:
                print("Testing phrase:  [\"{}\"] ".format(" ".join(joke_instance)))
                print("*" * 50)

                with open(report_file, "a", encoding="utf-8") as f:
                    f.write(f'Prompt: {" ".join(joke_instance)}\n')

                for _ in range(max_gen):
                    text_to_nums = []
                    for x in joke_instance:
                        text_to_nums.append(self.find_word_number(x))
                    text_to_nums = torch.tensor(text_to_nums, dtype=torch.long).unsqueeze(0)

                    text_to_nums = text_to_nums.to(device)

                    output,_ = self.forward(text_to_nums)

                    last_pred = output[0,-1]

                    # make unknown words very unlikely to be picked
                    last_pred[self.word_bank["<UNK>"]] = -1e9

                    # Higher randomness (1 < randomness) makes weights more similar
                    # Lower randomness (1 > randomness) makes weights less similar
                    randomness = 0.75
                    probs = torch.softmax(last_pred / randomness, dim=0)
                    # multinomial randomly selects between words based on the weight of that words class
                    next_word = torch.multinomial(probs, num_samples=1).item()

                    #next_word = torch.argmax(last_pred).item()
                    word = self.match_number_to_word(next_word)

                    if word == "<END>":
                        break

                    joke_instance.append(word)

                generated = " ".join(joke_instance)

                print(generated)


                with open(report_file, "a", encoding="utf-8") as f:
                    f.write(f"Generated: {generated}\n")
                    f.write("-" * 50 + "\n\n")


    def match_number_to_word(self, key):
        for word in self.word_bank:
            if self.word_bank[word] == key:
                #print(word)
                return word
        print("ERROR")
        return "<PAD"


    def start(self):
        self.training_process()


    def make_tensor_data(self,dataset):
        text_embedding_numbers_list = []

        # creates a tensor array for each word embedding number. The <END> index is applied at the end of each joke
        for sentence in dataset:
            if len(sentence) >= self.max_sentence_size - 1:
                text_embedding_numbers = []
                sentence = sentence[:self.max_sentence_size - 1]
                for word in sentence:
                    text_embedding_numbers.append(self.find_word_number(word))
                text_embedding_numbers.append(self.word_bank["<END>"])

                text_embedding_numbers_list.append(text_embedding_numbers)

            else:
                text_embedding_numbers = []
                entries_to_fill = (self.max_sentence_size - 1) - len(sentence)
                for word in sentence:
                    text_embedding_numbers.append(self.find_word_number(word))
                text_embedding_numbers.append(self.word_bank["<END>"])
                for i in range(entries_to_fill):
                    text_embedding_numbers.append(0)

                text_embedding_numbers_list.append(text_embedding_numbers)

        text_embedding_numbers_list = torch.tensor(text_embedding_numbers_list, dtype=torch.long)

        return text_embedding_numbers_list


    def find_word_number(self,word):
        if word not in self.word_bank:
            return self.word_bank["<UNK>"]
        else:
            return self.word_bank[word]


    def load_glove_embedding(self):
        print("Adjusting embeddings and creating word bank...")
        vocab_word = []
        embedding_number = []

        list_of_embeddings = []

        # Add special vocab characters before reading gloVe data
        vocab_word.append("<PAD>")
        embedding_number.append(0)
        x = []
        for i in range(self.embedding_size):
            x.append(0)
        list_of_embeddings.append(x)

        vocab_word.append("<UNK>")
        embedding_number.append(1)
        x = []
        for i in range(self.embedding_size):
            x.append(random.uniform(-0.1, 0.1))
        list_of_embeddings.append(x)

        vocab_word.append("<END>")
        embedding_number.append(2)
        x = []
        for i in range(self.embedding_size):
            x.append(random.uniform(-0.1, 0.1))
        list_of_embeddings.append(x)

        words_added = 3

        with open("data/joke_generation_data/gloVe.txt",  'r', encoding='utf-8') as file:
            print("Loading gloVE embeddings and copying...")
            for line in file:
                embedding_values = []
                for j,entry in enumerate(line.split()):
                    if j == 0:
                        #print(entry)
                        vocab_word.append(entry)
                    else:
                        try:
                            embedding_values.append(float(entry))
                        except ValueError:
                            # this is just to fix the error that the word ". . ." causes
                            vocab_word[-1] = vocab_word[-1] + entry
                list_of_embeddings.append(embedding_values)
                embedding_number.append(words_added)
                words_added += 1

                if words_added >= self.vocab_size:
                    break

        self.embedding_bank = torch.tensor(list_of_embeddings, dtype=torch.float)
        print("Embeddings updated...")
        self.word_bank = dict(zip(vocab_word, embedding_number))
        print("Word bank updated...")
        #print(self.word_bank)
        #print(pretrained_embedding_data)
