from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/.255,
                                   validation_split = 0.2,
                                   )
test_datagen = ImageDataGenerator(rescale = 1/.255,
                                  )

target = (128,128)

train_data = train_datagen.flow_from_directory('./covid-data/train/',
                                               target_size = target,
                                               color_mode = 'grayscale',
                                               class_mode = 'categorical',
                                               batch_size = 32,
                                               subset = 'training')
valid_data = train_datagen.flow_from_directory('./covid-data/train/',
                                               target_size = target,
                                               color_mode = 'grayscale',
                                               class_mode = 'categorical',
                                               batch_size = 32,
                                               subset = 'validation')
test_data = test_datagen.flow_from_directory('./covid-data/test/',
                                               target_size = target,
                                               color_mode = 'grayscale',
                                               class_mode = 'categorical',
                                               batch_size = 32)

import keras
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (128,128,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(train_data, validation_data = valid_data, epochs = 30)

model.evaluate(test_data)
