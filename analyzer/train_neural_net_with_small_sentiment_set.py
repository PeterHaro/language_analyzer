import math

import pandas as pd
import tensorflow as tf
from transformers import AutoConfig, TFAutoModelForSequenceClassification, optimization_tf, \
    AutoTokenizer, pipeline

model_name = 'NbAiLab/nb-bert-large'  # @param ["NbAiLab/nb-bert-base", "NbAiLab/nb-bert-large", "bert-base-multilingual-cased"]
batch_size = 16
init_lr = 2e-5
end_lr = 0
warmup_proportion = 0.1
num_epochs = 5
max_seq_length = 256

train_data = pd.read_csv(
    'https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/train.csv',
    names=["label", "text"]
)
dev_data = pd.read_csv(
    'https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/dev.csv',
    names=["label", "text"]
)
test_data = pd.read_csv(
    'https://raw.githubusercontent.com/ltgoslo/NorBERT/main/benchmarking/data/sentiment/no/test.csv',
    names=["label", "text"]
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Turn text into tokens
train_encodings = tokenizer(
    train_data["text"].tolist(), truncation=True, padding=True, max_length=max_seq_length
)
dev_encodings = tokenizer(
    dev_data["text"].tolist(), truncation=True, padding=True, max_length=max_seq_length
)
test_encodings = tokenizer(
    test_data["text"].tolist(), truncation=True, padding=True, max_length=max_seq_length
)

# Create a tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings), train_data["label"].tolist()
)).shuffle(1000).batch(batch_size)
dev_dataset = tf.data.Dataset.from_tensor_slices((
    dict(dev_encodings), dev_data["label"].tolist()
)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings), test_data["label"].tolist()
)).batch(batch_size)

print(
    f'The dataset is imported.\n\nThe training dataset has {len(train_data)} items.\nThe development dataset has {len(dev_data)} items. \nThe test dataset has {len(test_data)} items')
steps = math.ceil(len(train_data) / batch_size)
num_warmup_steps = round(steps * warmup_proportion * num_epochs)
print(
    f'You are planning to train for a total of {steps} steps * {num_epochs} epochs = {num_epochs * steps} steps. Warmup is {num_warmup_steps}, {round(100 * num_warmup_steps / (steps * num_epochs))}%. We recommend at least 10%.')

# Estimate the number of training steps
train_steps_per_epoch = int(len(train_dataset) / batch_size)
num_train_steps = train_steps_per_epoch * num_epochs

# Initialise a Model for Sequence Classification with 2 labels
config = AutoConfig.from_pretrained(model_name, num_labels=2)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Creating a scheduler gives us a bit more control
optimizer, lr_schedule = optimization_tf.create_optimizer(
    init_lr=init_lr,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps
)

# Compile the model
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])  # can also use any keras loss fn

# Start training
history = model.fit(train_dataset, validation_data=dev_dataset, epochs=num_epochs, batch_size=batch_size)

print(f'\nThe training has finished training after {num_epochs} epochs.')

# Ucomment the line under to save the model
# model.save_weights("/content/mymodel.h5")
sentiment_classifier = pipeline('sentiment-analysis', model=model, device=0)
candidate_sentances = ["Jeg liker å gå på stranden", "Jeg hater å gå på stranden", "Jeg elsker å hate de som hater"]
sentiment_results = sentiment_classifier(candidate_sentances, multi_class=True)
print(sentiment_results)
