
class DataBaseCreation:

    def __init__(self, array_of_strings, labels):

        self.y = labels
        self.x = []

        self.number_of_times_appears = {}

        max_line = -1
        min_line = 10000

        for line in array_of_strings:

            words = line.split()

            for word in words:

                if word in self.number_of_times_appears:
                    self.number_of_times_appears[word] += 1
                else:
                    self.number_of_times_appears[word] = 1

        self.word_to_id = {"~~unown~~": 0}

        for line in array_of_strings:

            words = line.split()
            arr = []

            for word in words:

                if word not in self.number_of_times_appears:
                    continue

                if self.number_of_times_appears[word] < 10:

                    continue

                if word not in self.word_to_id:

                    length = len(self.word_to_id)
                    self.word_to_id[word] = length
                    self.maxIndex = length + 1

                arr.append(self.word_to_id[word])

            while len(arr) < 62:
                arr.append(0)

            arr = arr[:62]

            self.x.append(arr)
            max_line = max(max_line, len(arr))
            min_line = min(min_line, len(arr))

            if len(arr) == 1:
                print(line)

        print(max_line)
        print(min_line)


    def get_x(self, array_of_strings):

        to_return = []

        for line in array_of_strings:

            words = line.split()
            arr = []

            for word in words:

                if word not in self.number_of_times_appears:
                    continue

                if self.number_of_times_appears[word] < 10:
                    continue

                if word not in self.word_to_id:
                    arr.append(0)
                else:
                    arr.append(self.word_to_id[word])

            while len(arr) < 62:
                arr.append(0)

            arr = arr[:62]

            to_return.append(arr)

        return to_return


