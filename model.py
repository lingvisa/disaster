from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import tensorflow_datasets as tfds

train_file_path = "./train.csv"
test_file_path = "./test.csv"

LABEL_COLUMN = 'target'
CSV_COLUMNS1 = ['text', 'target']
CSV_COLUMNS2 = ['text', 'target']

def get_dataset(file_path, CSV_COLUMNS, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=32,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        select_columns=CSV_COLUMNS,
        **kwargs)
    return dataset

sample_submission = pd.read_csv("./sample_submission.csv")

raw_train_data = get_dataset(train_file_path, CSV_COLUMNS1)
raw_test_data = get_dataset(test_file_path, CSV_COLUMNS2)

def extract_train_tensor(example, label):
    return example['text'], label

test_data = raw_test_data.map(lambda ex, label: extract_train_tensor(ex, label))
test_data_size = len(list(test_data))
print("test size: ", test_data_size)

train_data_all = raw_train_data.map(lambda ex, label: extract_train_tensor(ex, label))
val_data_size = len(list(train_data_all))
#
train_data_size = len(list(train_data_all))
print("train all size: ", train_data_size)
# train data = 200 * 32 = 6400
train_size = 200
train_data = train_data_all.take(train_size)
val_data = train_data_all.skip(train_size)
t_size = len(list(train_data))
v_size = len(list(val_data))
print("train size: {} val size: {}".format(t_size, v_size))

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_data,
                    epochs=20,
                    validation_data=val_data,
                    verbose=1)

results = model.evaluate(test_data, verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

import numpy as np

predictions = model.predict(test_data)
predictions = np.where(predictions > 0.5, 1, 0)
sample_submission['target'] = predictions
print(predictions)


sample_submission.to_csv("submission.csv", index=False)
