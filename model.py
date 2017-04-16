import csv
import cv2
import numpy as np
import sklearn
import random

def process_image(fileline):
  filepath = './data/IMG/' + fileline.split("/")[-1]
  image = cv2.imread(filepath)
  return image

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: #loop forever so the generator never stops
    random.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      measurements = []

      center_image = process_image(line[0])
      left_image = process_image(line[1])
      right_image = process_image(line[2])

      center_measurement = float(line[3])
      correction = 0.2
      left_measurement = center_measurement + correction
      right_measurement = center_measurement - correction

      flipped_center_image = np.fliplr(center_image)
      flipped_left_image = np.fliplr(left_image)
      flipped_right_image = np.fliplr(right_image)

      flipped_center_measurement = -center_measurement
      flipped_left_measurement = -left_measurement
      flipped_right_measurement = -right_measurement

      images.extend([center_image, left_image, right_image, flipped_center_image, flipped_left_image, flipped_right_image])
      measurements.extend([center_measurement, left_measurement, right_measurement, flipped_center_measurement, flipped_left_measurement, flipped_right_measurement])

    X_train = np.array(images)
    y_train = np.array(measurements)
    yield sklearn.utils.shuffle(X_train, y_train)

samples = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
    # print("this is line: ", line[0])

# using generators means we have to manually split train/validation sets and shuffle
# the data, since we can no longer use Keras's validation_split and shuffle methods
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# model training
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

row, col, ch = 160, 320, 3

model = Sequential()

# preprocessing, normalizing (/255) and mean centering (-.5)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))

model.add(Cropping2D(cropping=((60,25), (0,0))))

# 1st CNN layer
nb_filters =24
filter_size = (5,5)
strides = (2,2)
model.add(Convolution2D(nb_filters, filter_size[0], filter_size[1], subsample=strides, border_mode='valid', activation="relu"))
model.add(Dropout(0.5))

# 2nd CNN layer
nb_filters_2 = 36
filter_size_2 = (5,5)
strides_2 = (2,2)
model.add(Convolution2D(nb_filters_2, filter_size_2[0], filter_size_2[1], subsample=strides_2, border_mode='valid', activation="relu"))
model.add(Dropout(0.5))

# 3rd CNN layer
nb_filters_3 = 48
filter_size_3 = (5,5)
strides_3 = (2,2)
model.add(Convolution2D(nb_filters_3, filter_size_3[0], filter_size_3[1], subsample=strides_3, border_mode='valid', activation="relu"))
model.add(Dropout(0.5))

# 4th CNN layer
nb_filters_4 =64
filter_size_4 = (3,3)
model.add(Convolution2D(nb_filters_4, filter_size_4[0], filter_size_4[1], border_mode='valid', activation="relu"))
model.add(Dropout(0.5))

# 5th CNN layer
nb_filters_5 =64
filter_size_5 = (3,3)
model.add(Convolution2D(nb_filters_5, filter_size_5[0], filter_size_5[1], border_mode='valid', activation="relu"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) #single node representing steering angle, unlike classification, which has # of final nodes equal to number of classes

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
