import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data
import matplotlib.pyplot as plt
import random
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import json
import os
from BiLSTM import BiLSTM

# Useful Link:
# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
def accuracy(probs, target):
    winners = probs.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy

def cleanup_text(texts):
    cleaned_text = []
    with open('contractions.json', 'r') as fp:
        contractions = json.load(fp)

    for i, tok in enumerate(texts):
        new_tok = tok.lower()
        if new_tok in contractions:
            for word in contractions[new_tok].split():
                cleaned_text.append(word)
        else:
            if new_tok.endswith("'s"):
                new_tok = new_tok[:-2]
            cleaned_text.append(new_tok)
    return cleaned_text

def create_iterator(train_data, valid_data, test_data, batch_size):
    #  BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    # by setting sort_within_batch = True.
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text), # Sort the batches by text length size
        sort_within_batch = True)
    return train_iterator, valid_iterator, test_iterator


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        # retrieve text and no. of words
        text, text_lengths = batch.text

        predictions = model(text, text_lengths)

        # loss = criterion(predictions.squeeze(1), batch.score.type_as(predictions))
        loss = criterion(predictions.squeeze(1), batch.score)

        # print(predictions.shape)
        # print(batch.score.shape)
        # print(predictions)
        # print(batch.text.shape)
        # print(batch.score)

        acc = accuracy(predictions, batch.score)

        # perform backpropagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths)

            loss = criterion(predictions.squeeze(), batch.score)

            
            acc = accuracy(predictions, batch.score)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict(model, sentence, text):
    tokenized = cleanup_text(sentence)  #tokenize the sentence 
    indexed = [text.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed)#.to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()

def run_train(epochs, model, train_iterator, valid_iterator, optimizer, criterion, model_type):
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights'+'_'+model_type+'.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def plot_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].set_title('Model Accuracy')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    fig.tight_layout()
    plt.show()

def main():
    train = True
    lr = 1e-3
    batch_size = 50
    dropout_keep_prob = 0.5
    max_document_length = 100  # each sentence has until 100 words
    dev_size = 0.9 # split percentage to train\validation data
    seed = 1
    num_hidden_nodes = 64
    hidden_dim2 = 128
    num_layers = 2  # LSTM layers
    num_epochs = 10

    text = Field(sequential=True, use_vocab=True, preprocessing=cleanup_text, lower=True, batch_first=True, include_lengths=True)
    score = Field(sequential=False, use_vocab = False, dtype=torch.float, batch_first=True)
    
    fields = {"text": ("text", text), "score": ("score",score)}

    train_data, test_data = TabularDataset.splits(
    path="../data/", train="train.csv", test="test.csv", format="csv", fields=fields
    )

    train_data, valid_data = train_data.split(split_ratio=dev_size, random_state=random.seed(seed))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    text.build_vocab(train_data,vectors="glove.6B.300d")
    train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=32,
    )
    # print(train_data.dir())
    train_iterator, valid_iterator, test_iterator = create_iterator(train_data, valid_data, test_data, batch_size)

    model = BiLSTM(len(text.vocab), num_hidden_nodes, hidden_dim2 , num_layers, dropout_keep_prob)
    
    print(text.vocab.vectors)

    #No. of unique tokens in label
    print("Size of vocabulary:",len(text.vocab))

    #Commonly used words
    print("Commonly used words:",text.vocab.freqs.most_common(10))

    #Word dictionary
    # print("Word dictionary:",text.vocab.stoi)

    model.embedding.weight.data.copy_(text.vocab.vectors)
    
    loss = nn.MSELoss()

    # optimization algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train and evaluation
    if (train):
        # train and evaluation
        run_train(num_epochs, model, train_iterator, valid_iterator, optimizer, loss, 'LSTM')

        # load weights
    model.load_state_dict(torch.load(os.path.join('./', "saved_weights_LSTM.pt")))
    # predict
    test_loss, test_acc = evaluate(model, test_iterator, loss)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    sentence = "Tammy is playing football outside."
    print(predict(model, sentence, text))

if __name__ == '__main__':
    main()