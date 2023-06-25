import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, AveragePooling2D, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

class AI_Modelo:
   

    def create_model_categorial(self, num_classes=7):
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
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self


    def create_model_binary(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, activation='relu', input_shape=(256,256,3)))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(AveragePooling2D())
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
        return self
    

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        return self
    

    def run(self, train_data, validation_data, test_data, epochs=25):
        hist = self.model.fit(train_data, validation_data=validation_data, epochs=epochs)
        return hist
    

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


    def predict(self, data):
        return self.model.predict(data)
    

    def save_model(self, output = os.path.join(os.getcwd(), "models", "model.h5")):
        self.model.save(output)