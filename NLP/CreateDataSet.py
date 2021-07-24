
from DataReader import DataDirector


class CreateDataSet:

    def __init__(self):

        self.data_director = DataDirector()

        self.x, self.y = self.data_director.get_X_Y(1000, 50)


#create_data_set = CreateDataSet()

