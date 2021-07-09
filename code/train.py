import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data
import matplotlib.pyplot as plt
import random
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torchtext.vocab as vocab
import json
import os
from BiLSTM import BiLSTM

# Useful Link:
# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial1.py
# https://github.com/galhev/Neural-Sentiment-Analyzer-for-Modern-Hebrew/blob/8e247c5dc41b8d2e5a9b41581273c426575e4708/models/lstm_model.py#L5
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

def use_number(texts):
    for i,text in enumerate(texts):
        texts[i] = texts[i].replace('[','')
        texts[i] = texts[i].replace(']','')
        texts[i] = texts[i].replace(',','' )
    return texts


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
        skills = batch.skills

        predictions = model(text, text_lengths,skills)

        # loss = criterion(predictions.squeeze(1), batch.score.type_as(predictions))
        loss = criterion(predictions.squeeze(1), batch.score)

        acc = accuracy(predictions, batch.score)

        # perform backpropagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator)#, epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            skills = batch.skills

            predictions = model(text, text_lengths,skills)

            loss = criterion(predictions.squeeze(), batch.score)

            
            acc = accuracy(predictions, batch.score)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator)#, epoch_acc / len(iterator)

def predict(model, sentence, text, skills):
    tokenized = cleanup_text(sentence)  #tokenize the sentence 
    indexed = [text.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed)#.to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()

def run_train(epochs, model, train_iterator, valid_iterator, optimizer, criterion, grade_name):
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        print('epoch:', epoch)
        # train the model
        train_loss = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights'+'_'+grade_name+'.pt')

        print(f'\nTrain Loss: {train_loss:.3f}')# | Train Acc: {train_acc * 100:.2f}%')
        print(f'Validation Loss: {valid_loss:.3f}')# |  Val. Acc: {valid_acc * 100:.2f}%')


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
    batch_size = 5000
    dropout_keep_prob = 0.1
    # max_document_length = 100  # each sentence has until 100 words
    dev_size = 0.9 # split percentage to train\validation data
    seed = 1
    num_hidden_nodes = 64
    hidden_dim2 = 128
    skill_dim = 32
    num_layers = 1  # LSTM layers
    num_epochs = 9
    grade_name = "gr3"
    print(f"{grade_name} model\n")

    text = Field(sequential=True, use_vocab=True, fix_length=500,preprocessing=cleanup_text, lower=True, batch_first=True, include_lengths=True)
    score = Field(sequential=False, use_vocab = False, dtype=torch.float, batch_first=True)
    skills = Field(sequential=True, preprocessing=use_number, use_vocab = True, batch_first=True)
    
    fields = {"text": ("text", text), "score": ("score",score), "skills": ("skills",skills)}

    train_data, test_data = TabularDataset.splits(
    path="../data/"+grade_name, train="train.csv", test="test.csv", format="csv", fields=fields
    )

    train_data, valid_data = train_data.split(split_ratio=dev_size, random_state=random.seed(seed))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    # Build vocab embedding for input sentences
    text.build_vocab(train_data,vectors="glove.6B.300d")

    # Build number representatoin for input skill matrix
    # https://github.com/pytorch/text/issues/722
    number_embed = vocab.Vectors(name = 'number.txt')
    skills.build_vocab(train_data, vectors=number_embed)
    

    train_iterator, valid_iterator, test_iterator = create_iterator(train_data, valid_data, test_data, batch_size)

    model = BiLSTM(len(text.vocab), num_hidden_nodes, hidden_dim2 , num_layers, dropout_keep_prob,len(skills.vocab), skill_dim)

    #No. of unique tokens in label
    print("Size of vocabulary:",len(text.vocab))

    #Commonly used words
    print("Commonly used words:",text.vocab.freqs.most_common(10))

    #Word dictionary
    # print("Word dictionary:",text.vocab.stoi)

    model.embedding.weight.data.copy_(text.vocab.vectors)

    model.number_embed.weight.data.copy_(skills.vocab.vectors)
    model.number_embed.weight.requires_grad = False # freeze this layer to avoid mapping being changed
    # https://www.oreilly.com/library/view/deep-learning-with/9781788624336/fa953b7d-daac-4c4e-be74-66a7e94de98e.xhtml
    
    loss = nn.MSELoss()

    # optimization algorithm
    optimizer = torch.optim.Adam([ param for param in model.parameters() if param.requires_grad == True], lr=lr)

    # train and evaluation
    if (train):
        # train and evaluation
        run_train(num_epochs, model, train_iterator, valid_iterator, optimizer, loss, grade_name)

        # load weights
    model.load_state_dict(torch.load(os.path.join('./', 'saved_weights'+'_'+grade_name+'.pt')))
    # predict
    test_loss = evaluate(model, test_iterator, loss)
    print(f'Test Loss: {test_loss:.3f}')
    # sentence = "I come to dance class early, to stretch in the quiet of the room. My body quivers, thinking of tonight. The class begins. I exercise hard, till sweat streams down my face and back. Music fills the room. We do our leaps across the floor. We want to dance full out, but our teacher tells us. Now we'll go home to rest and prepare, because tonight, there is a performance."
    # print(predict(model, sentence, text))

if __name__ == '__main__':
    main()