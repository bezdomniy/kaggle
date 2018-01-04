from model_cnn import *
from voice import load_test_data
import csv



def run_training(validation=True):
    spectrograms,cats,lengths = load_data(limited=True)
    spectrograms,cats,lengths = shuffle(spectrograms,cats,lengths, random_state=0)

    train_data, train_lengths, train_labels = getData(spectrograms,cats,lengths)
    train(train_data, train_lengths, train_labels, lim=True)

    if validation:
        val_data, val_lengths, val_labels = getData(spectrograms,cats,lengths, test=True)
        validate(val_data, val_lengths, val_labels)

def run_testing():
    spectrograms, lengths, file_list = load_test_data()
    predictions = test(spectrograms, lengths)
    #predictions = test(1, 1)
    
#run_training()
run_testing()