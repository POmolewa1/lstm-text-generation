import os
import csv


filter = [",", ".", "!", "<", ">", "'", '"', "(", ")", "/", ":", ";", "?", "-", "_", "*"]
filter2 = ["<", ">", "(", ")", "/", ":", "_", "*"]

def create_joke_data(datapath):
    jokes_list = []

    with open(datapath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentence = []
            for word in row["Joke"].split():
                if in_filter(word):
                    separate_from_filter(word, sentence)
                else:
                    sentence.append(tokenize(word,sentence))
            jokes_list.append(sentence)

    return jokes_list

def create_dataset(datapath):
    print("Creating dataset...")
    labels = []
    full_review = []
    print("Reading negative reviews...")
    with os.scandir((datapath + "/neg")) as folder:
        for file in folder:
            sentence = []
            with open(file.path, 'r', encoding='utf-8') as text_file:
                for line in text_file:
                    for word in line.split():
                        if in_filter(word):
                            separate_from_filter(word, sentence)
                        else:
                            token_word = tokenize(word,sentence)
                            sentence.append(token_word)

            #print(sentence)
            full_review.append(sentence)
            labels.append(0)

    print("Reading positive reviews...")
    with os.scandir((datapath + "/pos")) as folder:
        for file in folder:
            sentence = []
            with open(file.path, 'r', encoding='utf-8') as text_file:
                for line in text_file:
                    for word in line.split():
                        if in_filter(word):
                            separate_from_filter(word, sentence)
                        else:
                            token_word = tokenize(word,sentence)
                            sentence.append(token_word)

            #print(sentence)
            full_review.append(sentence)
            labels.append(1)

    counter = 0
    with open("negative.txt", 'w', encoding='utf-8') as file:
        for text in full_review:
     #       print(text)
            file.write(" ".join(text) + str(labels[counter]) + '\n')
            #print("\n")
            counter += 1
            if counter == 20:
                break

    sections = ["text", "label"]
    print("Dataset created...")
    return dict(zip(sections, [full_review, labels]))


def tokenize(word,sentence_array):
    token = []

    if has_quotes(word):
        return word


    for i,letter in enumerate(word):
        #if letter.isalpha():
        if letter not in filter:
            if letter.isupper():
                letter = letter.lower()
                token.append(str(letter))
            else:
                token.append(letter)
        else:
            if len(token) > 0:
                sentence_array.append("".join(token))
            #sentence_array.append(letter)
            return tokenize(word[i+1:],sentence_array)

    token = "".join(token)

    return token

def in_filter(word):
    quotes = False
    for letter in word:
        if letter in filter:
            quotes = True
            break

    return quotes

def has_quotes(word):
    quotes = False
    for letter in word:
        if letter == "'" or letter == '"':
        #if letter in filter:
            quotes = True
            break

    return quotes

def separate_from_filter(word, sentence_array):
    if word == '':
        return
    # If the word has quotes separately add the quotes and the word ex -> "hi"  [' , hi, '] does not append characters in filter 2
    if word[0] in filter:
        if word[0] == word[-1]:
            sentence_array.append(word[0])
            separate_from_filter(word[1:-1], sentence_array)
            #token,_ = tokenize(word[1:-1])
            #sentence_array.append(token)
            sentence_array.append(word[-1])
        else:
            if word[0] not in filter2:
                sentence_array.append(word[0])
            separate_from_filter(word[1:], sentence_array)
        return
    elif word[-1] in filter:
        separate_from_filter(word[:-1], sentence_array)
        if word[0] not in filter2:
            sentence_array.append(word[-1])
        return

    # If the word still has quotes it's an abbreviation
    if has_quotes(word):
        quote_index = 0
        for i,letter in enumerate(word):
            if letter == "'":
                quote_index = i
                break

        # words like don't need to be separated into [do, n't]
        if word[quote_index - 1] == "n":
            #sentence_array.append(word[0: quote_index - 1])
            separate_from_filter(word[0: quote_index - 1], sentence_array)
            #print(word[0: quote_index - 1])
            sentence_array.append(word[quote_index - 1:])
            #separate_from_filter(word[quote_index - 1:], sentence_array)
            #print(word[quote_index - 1:])

        else:
            #sentence_array.append(word[0: quote_index])
            separate_from_filter(word[0: quote_index], sentence_array)
            #print(word[0: quote_index])
            sentence_array.append(word[quote_index:])
            #print(word[quote_index:])
        return
    #print(word)
    # gets rid of the <br> that is in some files
    if word != "br":
        sentence_array.append(tokenize(word,sentence_array))