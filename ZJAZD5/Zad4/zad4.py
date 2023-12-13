from silence_tensorflow import silence_tensorflow

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling

silence_tensorflow()

train_ds = tf.keras.utils.image_dataset_from_directory(
  'horse-or-human',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(300, 300),
  batch_size=100)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'validation-horse-or-human',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(300, 300),
  batch_size=100)

class_names = train_ds.class_names

model = Sequential()
model.add(Rescaling(1./255, input_shape=(300, 300, 3)))
model.add(Conv2D(16, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=10, validation_data=val_ds)

loss, accuracy = model.evaluate(val_ds, verbose=0)

print("Test accuracy:", accuracy)
