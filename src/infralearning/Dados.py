import tensorflow as tf


class Dados():

    def __init__(self, 
                 data_dir, 
                 train_percentage = 0.95, 
                 valid_percentage = 0.03, 
                 test_percentage = 0.02):
        self.data = (tf.keras.utils.image_dataset_from_directory(data_dir,batch_size=64)).map(lambda x, y: (x/255, y))
        self.num_classes = self.get_num_classes()
        self.setup_lists(train_percentage, valid_percentage, test_percentage)


    def get_num_classes(self):
        labels = []
        for _, label in self.data:
            labels += label.numpy().tolist()
        return len(set(labels))


    def setup_lists(self, train_percentage, valid_percentage, test_percentage):
        num_of_batches = len(self.data)
        train_size = int(train_percentage * num_of_batches)
        valid_size = int(valid_percentage * num_of_batches)
        test_size = int(test_percentage * num_of_batches) + (num_of_batches - train_size - valid_size - int(test_percentage * num_of_batches))

        print(num_of_batches, "batches in [ Training=", train_size, "  Test=", test_size, "  Validation=", valid_size, "]")
        assert train_percentage + valid_percentage + test_percentage == 1, "Percentages must sum to 1"
        assert (train_size + valid_size + test_size) == num_of_batches, "Splitting Batches error"

        self.train = self.data.take(train_size)
        self.validation = self.data.skip(train_size).take(valid_size)
        self.test = self.data.skip(train_size+valid_size).take(test_size)

        if (self.num_classes > 2):
            self.train = self.train.map(lambda x, y: (x, tf.one_hot(y, self.num_classes)))
            self.validation = self.validation.map(lambda x, y: (x, tf.one_hot(y, self.num_classes)))
            self.test = self.test.map(lambda x, y: (x, tf.one_hot(y, self.num_classes)))