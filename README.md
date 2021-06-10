# Predicting Reading Comprehension Score for Elementry Students
## Usage
* `BiLSTM.py` contains the bidirecional LSTM model designed for the prediction task.
* `train.py` includes all functions to train, evaluate, predict with the BiLSTM Model
* `generate_dataset.py` reads the raw data and generate dataset with "text" and "score" for the prediction task.
* `split_dataset.py` is a utility function to split the dataset into training and testing set with a given percentage.
* `tokenize_word_embed.py` is a testing script for developing tokenizers and check embedding coverage on tokenized text.
* `create_embed.py` generates a embedding pickle file that indexes vocab of a given corpus and appends GloVe embedding to each word.
