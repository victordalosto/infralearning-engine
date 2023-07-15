import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

from infralearning.Dados import Dados


class AI_Modelo:

    def create_model_binary(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=16, kernel_size=(3,3), strides=1, activation='relu', input_shape=(256,256,3)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=48, kernel_size=(3,3), strides=1, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), strides=1, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
        return self
   

    def create_model_categorial(self, dados:Dados):
        self.model = Sequential()
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', input_shape=(256, 256, 3)))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(dados.num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self
    

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        return self
    

    def run(self, dados:Dados, epochs=25):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "logs"))
        self.hist = self.model.fit(dados.train, validation_data=dados.validation, epochs=epochs, callbacks=[tensorboard_callback])
    

    def show_precision(self, test):
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()
        for batch in test.as_numpy_iterator(): 
            X, y = batch
            yhat = self.model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)
        print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

    
    def plot_loss_graph(self):
        print("\n Plotting Loss Graphs")
        fig = plt.figure()
        plt.plot(self.hist.history['loss'], color='teal', label='loss')
        plt.plot(self.hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()


    def plot_accuracy_graph(self):
        print("\n Plotting Accuracy Graphs")
        fig = plt.figure()
        plt.plot(self.hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()


    def predict(self, data):
        return self.model.predict(data)
    

    def save_model(self, output = os.path.join(os.getcwd(), "models", "model.h5")):
        self.model.save(output)