# import regular packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from subprocess import check_output
import time

# import deep learning packages
from keras.models import Sequential
from keras.layers import Conv2D, Convolution2D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Activation, \
    MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint

# check what is in the directory
# print(check_output(["ls", "/Users/leeeyler/Documents/DATS_6203/Final_Project"]).decode('utf8'))


# time script execution
start = time.time()
print("executing script")

# load data
train_df = pd.read_json('train.json')
test_df = pd.read_json('test.json')

# what do the bands look like?
train_df.columns
train_df.head(1)
train_df['band_1'].head(1)
train_df['band_2'].head(1)

# let's view a few of the training images
# plt.imshow(np.array(train_df['band_1'][1]).reshape(75, 75))
# plt.imshow(np.array(train_df['band_1'][2]).reshape(75, 75))
# plt.imshow(np.array(train_df['band_1'][5]).reshape(75, 75))

# training data
# convert X vals from dataframe to array
# convert to float32 and rehape to 75x75
# concatenate the bands together
# convert y vals from dataframe to array
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train_df["is_iceberg"])
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# test data
# convert X vals from dataframe to array
# convert to float32 and rehape to 75x75
# concatenate the bands together
x_band1_test = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
x_band2_test = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
X_test = np.concatenate([x_band1_test[:, :, :, np.newaxis], x_band2_test[:, :, :, np.newaxis]], axis=-1)
print("X_test:", X_test.shape)

# this will be sequential CNN
model = Sequential()

# convolution layer 1
model.add(Conv2D(filters=128,kernel_size=(4,4),strides=1,\
            padding='same', activation='relu', input_shape=(75,75,2)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(.2))

# convolution layer 2
model.add(Conv2D(filters=128,kernel_size=(4,4),strides=1,\
            padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(.2))

# convolution layer 3
model.add(Conv2D(filters=256,kernel_size=(4,4),strides=1,\
            padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(.2))

# convolution layer 4
model.add(Conv2D(filters=256,kernel_size=(4,4),strides=1,\
            padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(.2))

# flatten for fully connected layers
model.add(Flatten())

# fully connected layer 1
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.2))

# fully connected layer 2
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.2))

# fully connected layer 3
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.2))

# output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

# define loss function and optimizer
adadelta = optimizers.Adadelta(lr=.10, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(adadelta, 'binary_crossentropy', metrics=['accuracy'])

# view model summary
model.summary()

# train model
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
hist = model.fit(X_train, y_train, validation_split=.2, epochs=500, batch_size=25, callbacks=[early_stopping],
                 verbose=1, shuffle = True)

plt.figure(1)
plt.plot(range(1, len(hist.history['val_acc']) + 1), hist.history['val_acc'], label='Line 1')
plt.xlabel('EPOCH')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy')
locator = matplotlib.ticker.MultipleLocator(1)
plt.gca().xaxis.set_major_locator(locator)
formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
plt.gca().xaxis.set_major_formatter(formatter)

plt.figure(2)
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'], label='Line 2')
plt.xlabel('EPOCH')
plt.ylabel('Validation Loss')
plt.title('Validation Loss')
locator = matplotlib.ticker.MultipleLocator(1)
plt.gca().xaxis.set_major_locator(locator)
formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
plt.gca().xaxis.set_major_formatter(formatter)


## make predictions
prediction = model.predict(X_test, verbose=1)

## submission dataframe
submit_df = pd.DataFrame({'id': test_df['id'], 'is_iceberg': prediction.flatten()})
submit_df.to_csv('final_project_sub_4CNN768_3FC768_22poolsize_44kernel_15patience.csv', index=False)


# time script execution
end = time.time()
print("execution time:  ")
print(end - start)
plt.show()

