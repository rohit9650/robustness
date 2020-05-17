

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import timeit
import keras
from skimage.color import gray2rgb, rgb2gray, label2rgb
from skimage.util import montage as montage2d

# np.random.seed(1337)

def GetnnModel(dataset_name):
  K.clear_session()
  batch_size = 128
  epochs = 2
  channel = 1
  if dataset_name == "MNIST":
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10
    channel = 3
    # the data, shuffled and split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784')
    # make each image color so lime_image works correctly
    X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))],0)
    y_vec = mnist.target.astype(np.uint8)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                        train_size=0.55)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = (x_train - 0.5) * 2
    x_test = (x_test - 0.5) * 2
    ## testing
    # x_train = x_train[:1024] 
    # y_train = y_train[:1024]
    ## to be removed 

  if dataset_name == "olivetti_faces":
    # input image dimensions
    img_rows, img_cols = 64, 64
    num_classes = 40

    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces()
    # make each image color so lime_image works correctly
    X_vec = np.stack([gray2rgb(iimg) for iimg in faces.data.reshape((-1, 64, 64))],0)
    y_vec = faces.target.astype(np.uint8)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                        train_size=0.80)
    channel = 3
    batch_size = 2
    epochs = 20
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

  if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channel, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channel, img_rows, img_cols)
    input_shape = (channel, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel)
    input_shape = (img_rows, img_cols, channel)


  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)


  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax')) 
  # ^ IMPORTANT: notice that the final softmax must be in its own layer 
  # if we want to target pre-softmax units

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  return model

