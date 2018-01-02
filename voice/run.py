from model_cnn import *
from voice import load_test_data



def run_training(validation=True):
    spectrograms,cats,lengths = load_data()
    spectrograms,cats,lengths = shuffle(spectrograms,cats,lengths, random_state=0)

    
    train_data, train_lengths, train_labels = getData(spectrograms,cats,lengths)
    train(train_data, train_lengths, train_labels)

    if validation:
        val_data, val_lengths, val_labels = getData(spectrograms,cats,lengths, test=True)
        test(val_data, val_lengths, val_labels)

def run_testing():
    spectrograms, lengths = load_test_data()
    
run_training()