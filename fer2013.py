#from __future__ import division

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.utils import plot_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# get the data
filname = 'fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

X, Y = getData(filname)
num_class = len(set(Y))

# To see number of training data point available for each label
def balance_class(Y):
    num_class = set(Y)
    count_class = {}
    for i in range(len(num_class)):
        count_class[i] = sum([1 for y in Y if y == i])
    return count_class

balance = balance_class(Y)

N, D = X.shape
X = X.reshape(N, 48, 48, 1)

# Split in  training set : validation set :  testing set in 80:10:10
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

batch_size = 128
#%%
from keras.layers import Input, Conv2D, Activation, Dense, BatchNormalization
from keras.layers import MaxPooling2D, Dropout, Flatten

inputs = Input((48, 48, 1), name='input')

# conv1
c = Conv2D(64, (3, 3), padding='same', name='conv1')(inputs)
c = Activation('relu', name='conv1_relu')(c)
c = BatchNormalization(name='conv1_bn')(c)
c = Dropout(0.25)(c)

# conv2
c = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv2')(c)
c = Activation('relu', name='conv2_relu')(c)
c = BatchNormalization(name='conv2_bn')(c)
c = MaxPooling2D(pool_size=(2, 2),padding='valid',name='conv2_pl')(c)
c = Dropout(0.25)(c)

# conv3
c = Conv2D(256, (3, 3), padding='same', name='conv3')(c)
c = Activation('relu', name='conv3_relu')(c)
c = BatchNormalization(name='conv3_bn')(c)
c = Dropout(0.25)(c)

# conv4
c = Conv2D(256, (3, 3), padding='same', strides=(2, 2), name='conv4')(c)
c = Activation('relu', name='conv4_relu')(c)
c = BatchNormalization(name='conv4_bn')(c)
c = MaxPooling2D(pool_size=(2, 2),padding='valid',name='conv4_pl')(c)
c = Dropout(0.25)(c)
c = Flatten()(c)

# out
num_class = 7
c = Dense(256, activation='relu', name='fc1')(c)
c = Dropout(0.25)(c)
c = Dense(num_class, name='fc2')(c)
outputs = c = Activation('softmax', name='out')(c)

#%%
# 通过 Model 类创建模型，Model 类在初始化的时候需要指定模型的输入和输出
model = Model(inputs=inputs, outputs=outputs)
model.summary()

optim = SGD(1e-4, momentum=0.99, nesterov=True)
loss = categorical_crossentropy

# 定义损失函数、优化函数和评测方法
model.compile(optim, loss, metrics=['accuracy'])
epochs= 700
#hist = model.fit(
#                 X_train, y_train,
#                 batch_size=batch_size,
#                 epochs=epochs, verbose=2,
#                 validation_split=0.1
#                 )
#print(hist.history)
#model.save_weights('fer2013CNN_700.h5')
#%%
# Evaluate
# Model will predict the probability values for 7 labels for a test image

model.load_weights('fer2013CNN.h5')
score = model.predict(X_test)

new_X = [ np.argmax(item) for item in score ]
y_test2 = [ np.argmax(item) for item in y_test]

# Calculating categorical accuracy taking label having highest probability
accuracy = [ (x==y) for x,y in zip(new_X,y_test2) ]
print(" Accuracy on Test set : " , np.mean(accuracy))
#plot_model(model, to_file='model.png')
