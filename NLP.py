import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model

tf.__version__

path_to_Text_File = r''

text = open(path_to_Text_File, 'r', encoding="utf8").read()

print(text[:1000])

vocab = sorted(set(text))
print(vocab)
len(vocab)

char_to_ind = {u:i for i, u in enumerate(vocab)}
print(char_to_ind)

ind_to_char = np.array(vocab)
print(ind_to_char)

encoded_text = np.array([char_to_ind[c] for c in text])
print(encoded_text)

sample = text[:20]
print(sample)

encoded_text[:20]
seq_len = 120

total_num_seq = len(text)//(seq_len+1)
print(total_num_seq)

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

for i in char_dataset.take(500):
     print(ind_to_char[i.numpy()])

sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt

dataset = sequences.map(create_seq_targets)

for input_txt, target_txt in  dataset.take(1):
    print(input_txt.numpy())
    print(''.join(ind_to_char[input_txt.numpy()]))
    print('\n')
    print(target_txt.numpy())
    # There is an extra whitespace!
    print(''.join(ind_to_char[target_txt.numpy()]))

batch_size = 128
buffer_size = 10000
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

vocab_size = len(vocab)
print(vocab_size)

embed_dim = 64
rnn_neurons = 1026

def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss) 
    return model

model = create_model(
  vocab_size = vocab_size,
  embed_dim=embed_dim,
  rnn_neurons=rnn_neurons,
  batch_size=batch_size)

model.summary()

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape)

print(example_batch_predictions)

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print(sampled_indices)

epochs = 30
model.fit(dataset,epochs=epochs)

model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)

model.load_weights('model.h5')

model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_seed,gen_size=1000,temp=1.0):

  num_generate = gen_size
  input_eval = [char_to_ind[s] for s in start_seed]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
 
  temperature = temp
  model.reset_states()

  for i in range(num_generate):

      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(ind_to_char[predicted_id])

  return (start_seed + ''.join(text_generated))

generator = " "
gen_size_value = 0
print(generate_text(model,generator,gen_size=gen_size_value))