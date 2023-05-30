
from utils.model import *
from utils.load_data import *
from config import config
from pickle import load
import random
from tensorflow.keras.callbacks import ModelCheckpoint

# Random seed
random.seed(1035)

# Load the training data
images_features_train, captions_train = load_image_features(config, 'train'), load_clean_data(config, 'train')

# Load the validation data
images_features_val, captions_val = load_image_features(config, 'dev'), load_clean_data(config, 'dev')

# Tokenization
tokenizer = create_tokenizer(captions_train)
max_len = max_length(captions_train)
print(f"Max length: {max_len}")
vocab_size = len(tokenizer.word_index) + 1
print(f"Train vocab size: {vocab_size}")

# Define the model
model = define_model(vocab_size, max_len)
print('RNN Model summary:')
print(model.summary())


# Train the model and save after each epoch
epochs = 20
batch_size = 64

# Calculate steps
steps_train = len(captions_train)//batch_size
if len(captions_train)%batch_size!=0:
    steps_train = steps_train+1
steps_val = len(captions_val)//batch_size
if len(captions_val)%batch_size!=0:
    steps_val = steps_val+1

# Checkpoints
model_save_path = "./model_data/model_RNN_epoch-{epoch:02d}_train_loss-{loss:.4f}_val_loss-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

# Display infos
print('steps_train: {}, steps_val: {}'.format(steps_train,steps_val))
print('Batch Size: {}'.format(batch_size))
print('Total Number of Epochs = {}'.format(epochs))

# Shuffle train data
ids_train = list(captions_train.keys())
random.shuffle(ids_train)
captions_train_shuffled = {_id: captions_train[_id] for _id in ids_train}
caps_train = captions_train_shuffled

# Create the train data generator
# returns [[img_features, text_features], out_word]
generator_train = data_generator(images_features_train, caps_train, tokenizer, max_length, batch_size, 1035)
# Create the validation data generator
# returns [[img_features, text_features], out_word]
generator_val = data_generator(images_features_val, captions_val, tokenizer, max_length, batch_size, 1035)

# Fit for one epoch
model.fit_generator(generator_train, epochs=epochs, steps_per_epoch=steps_train, validation_data=generator_val, validation_steps=steps_val, callbacks=callbacks, verbose=1)





