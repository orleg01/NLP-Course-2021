
import os
from os.path import join
import urllib3


class Book:

    def __init__(self, where_to_save, url, indexes):

        self.where_to_save = where_to_save
        self.url = url
        self.indexes = indexes

    def save_data(self):

        for i in range(1, self.indexes):

            try:

                target_url = self.url + str(i)
                http = urllib3.PoolManager()
                response = http.request('GET', target_url)
                data = response.data.decode('utf-8')
                find = '<p>'
                end = 'Previous Chapter'
                start_index = data.find(find)
                data = data[start_index:]
                end_index = data.find(end)
                data = data[:end_index]

                find = '<p>'
                start_index = data.find(find)
                data = data[start_index:]

                find = 'Next Chapter'
                start_index = data.find(find)
                data = data[:start_index]

                keep_going = True
                while keep_going:

                    start_index = data.find('<')
                    end_index = data.find('>')

                    if start_index == -1:
                        break

                    data = data[:start_index] + data[end_index + 1:]


                for item in ['.', '!', '?', ',', '…', "\"", "\'", ";" ,'  ']:

                    while keep_going:

                        start_index = data.find(item)

                        if start_index == -1:
                            break

                        if item != '  ':
                            data = data[:start_index] + " " + data[start_index + 1:]
                        else:
                            data = data[:start_index] + data[start_index + 1:]

                for item in ['”', '“', "”", '’', '~']:

                    while keep_going:

                        start_index = data.find(item)

                        if start_index == -1:
                            break

                        if item != '  ':
                            data = data[:start_index] + data[start_index + 1:]

                data = data.lower()

                with open(join(self.where_to_save, str(i) + ".txt"), "w") as file:

                    file.write(data)

                print(self.where_to_save, "  ", i)
            except:
                os.remove(join(self.where_to_save, str(i) + ".txt"))



Book('./AgainstTheGods', 'https://www.wuxiaworld.com/novel/against-the-gods/atg-chapter-', 300).\
    save_data()

Book('./EmperorDomination', 'https://www.wuxiaworld.com/novel/emperors-domination/emperor-chapter-', 300).\
    save_data()

Book('./MartialGodAsura', 'https://www.wuxiaworld.com/novel/martial-god-asura/mga-chapter-', 300).\
    save_data()
