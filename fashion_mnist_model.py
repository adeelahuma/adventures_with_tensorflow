import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import math
import os

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

num_train = metadata.splits['train'].num_examples
num_test = metadata.splits['test'].num_examples
print(" {} train examples".format(num_train))
print(" {} test examples".format(num_test))

train_data, test_data = data['train'], data['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train_data = train_data.map(normalize)
test_data = test_data.map(normalize)


train_data = train_data.cache()
test_data = test_data.cache()


## Build Model

input_layer = tf.keras.layers.Flatten(input_shape=(28,28,1))
hidden_layer = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)


model = tf.keras.Sequential([input_layer, hidden_layer, output_layer])
optim = tf.keras.optimizers.Adam(0.1)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics= ["accuracy"])

##
BATCH_SIZE = 32

train_data= train_data.cache().repeat().shuffle(num_train).batch(BATCH_SIZE)
test_data = test_data.cache().batch(BATCH_SIZE)

# training

model.fit(train_data, epochs=5, steps_per_epoch=math.ceil(num_train/BATCH_SIZE))

## evaluate model 

test_loss, test_accuracy = model.evaluate(test_data, steps=math.ceil(num_test/BATCH_SIZE))

print("Accuracy:  {}".format(test_accuracy))


## make prediction

for test_images, test_labels in test_data.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()

    predictions = model.predict(test_images)


print("prediction {}  | actual {} ".format(np.argmax(predictions[0]), test_labels[0]))

##evaluate one image

img = test_images[10]

print(img.shape)

img = np.array([img])

print(img.shape)

pred = model.predict(img)

print("pred {} | actutal {}".format(np.argmax(pred), test_labels[10]))


# checkpoint 

checkpoint_path = 'fashion_mnist/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                 save_weights_only=True, 
                                                 verbose=1)

# train with checkpoints 

model.fit(train_data, epochs=6, steps_per_epoch=math.ceil(num_train/BATCH_SIZE), callbacks=[cp_callback])

