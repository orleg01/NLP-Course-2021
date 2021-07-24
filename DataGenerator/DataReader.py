
import os
from os.path import join

against_the_god = 'C:\\Users\\97254\\Documents\\python_projects\\PlayGround\\DataGenerator\\AgainstTheGods'
emperor_domination = 'C:\\Users\\97254\\Documents\\python_projects\\PlayGround\\DataGenerator\\EmperorDomination'
martial_god_asura = 'C:\\Users\\97254\\Documents\\python_projects\\PlayGround\\DataGenerator\\MartialGodAsura'


def non_empty(string):

    return [x for x in string if x]


def get_array_of_words(string):

    string = string.replace("mga:", "")
    string = string.replace("chapter", "")
    string = string.replace("–", "")
    string = string.replace("-", "")
    string = string.replace("*", "")
    string = string.replace(":", "")
    string = string.replace("[", "")
    string = string.replace("]", "")

    for i in range(0, 10):
        string = string.replace(str(i), "")

    return non_empty(string.split(" "))


class DataReader:

    def __init__(self, file_location):

        self.arr = []

        for file in os.listdir(file_location):

            with open(join(file_location, file), "r") as _file:

                arr = get_array_of_words(_file.readline())
                self.arr.append(arr)

    def get_arr(self, length, jump):

        to_return = []

        for arr in self.arr:

            for index in (0, len(arr) - length, jump):

                to_return.append(arr[index: index + length])

            to_return.append(arr[-length:])

        return to_return


class DataDirector:

    def __init__(self):

        self.atg = DataReader(against_the_god)
        self.ed = DataReader(emperor_domination)
        self.mga = DataReader(martial_god_asura)

        self.word_to_id = {}

        temp = [self.atg, self.ed, self.mga]

        for book in temp:

            for page in book.arr:

                for word in page:

                    count = len(self.word_to_id)

                    if word not in self.word_to_id:

                        self.word_to_id[word] = count

        self.id_to_word = {value: key for key, value in self.word_to_id.items()}

    def get_X_Y(self, length, jump):

        X = []
        Y = []

        temp = self.atg.get_arr(length, jump)
        for arr in temp:

            for index in range(len(arr)):

                arr[index] = self.word_to_id[arr[index]]

            X.append(arr)
            Y.append(1)

        temp = self.ed.get_arr(length, jump)
        for arr in temp:

            for index in range(len(arr)):

                arr[index] = self.word_to_id[arr[index]]

            X.append(arr)
            Y.append(2)

        temp = self.mga.get_arr(length, jump)
        for arr in temp:

            for index in range(len(arr)):

                arr[index] = self.word_to_id[arr[index]]

            X.append(arr)
            Y.append(3)

        return X, Y

#data_director = DataDirector()

