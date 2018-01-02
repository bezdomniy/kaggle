from scipy.io.wavfile import read, write
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import webrtcvad
from struct import *

path = './train/audio/'
test_path = './test/'
max_len = 126

def get_data(file_name):
    sample_rate, sample = read(file_name)
    if len(sample) != 16000:
        tail = np.zeros(16000-len(sample),np.int16)
        sample = np.concatenate([sample,tail])
    
    return sample, sample_rate

def remove_silence(sample,sample_rate):
    vad = webrtcvad.Vad(3)
    split = np.split(sample,100)
    is_speech_array = []
    for frame in split:
        bin = pack('l'*(sample_rate * 10//1000),*frame)
        is_speech_array.append(vad.is_speech(bin,sample_rate))

    # Assume if recording has speech in first frame, then VAD is mistaken
    if is_speech_array.count(is_speech_array[0]) != len(is_speech_array):
        i=0
        while True:
            if is_speech_array[i] == True:
                is_speech_array[i] = False
            else:
                break
            i+=1

    speech_frames = np.where(is_speech_array)[0]

    if len(speech_frames) <= 2:
        return None

    first_speech_frame, last_speech_frame = speech_frames[0] , speech_frames[-1]

    trimmed_split = split[first_speech_frame:last_speech_frame+1]
    return np.concatenate(trimmed_split)

def make_spectrogram(sample, sample_rate):
    #freqs, times, spectrogram = signal.spectrogram(sample, sample_rate)

    freqs, times, spectrogram = signal.stft(sample,sample_rate)

    return np.log1p(np.abs(spectrogram.T)), freqs, times
    #return np.log(spectrogram.T.astype(np.float32) + 1e-10), freqs, times

def show_graph(sample,sample_rate,freqs, times, spec):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, len(sample)/sample_rate, len(sample)), sample)

    ax2 = fig.add_subplot(212)
    ax2.imshow(spec.T, aspect='auto', origin='lower', 
            extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_title('Spectrogram')
    ax2.set_ylabel('Freqs in Hz')
    ax2.set_xlabel('Seconds')
    plt.show()


def load_data():
    if not os.path.isfile('spectrograms.npy'):
        print("Loading data for the first time...")

        categories = [os.path.split(os.path.split(f)[0])[1] for f in glob.glob(path+"*/")][:-1]
        categories.append('silence')

        one_hot_categories = pd.get_dummies(categories)
        file_list = []

        for category in categories[:-1]:
            file_list.extend(glob.glob(path+category+'/*'))


        spectrograms = np.zeros([len(file_list),max_len,129], dtype=np.float32)
        labels = np.zeros([len(file_list),31], dtype=np.int8)
        lengths = np.zeros([len(file_list)], dtype=np.int32)

        for i in range(len(file_list)):
            f = file_list[i]
            cat = os.path.basename(os.path.dirname(f))
            labels[i,:] = np.array(one_hot_categories[cat])
            sample,sample_rate = get_data(f)
            trimmed = remove_silence(sample,sample_rate)

            if trimmed is None:
                print('silence at ',i)
                spec, freqs, times = make_spectrogram(sample, sample_rate)
                lengths[i] = spec.shape[0]
                labels[i,:] = np.array(one_hot_categories['silence'])
                continue

            spec, freqs, times = make_spectrogram(trimmed, sample_rate)
            lengths[i] = spec.shape[0]

            spectrograms[i,:lengths[i],:] = spec

        np.save('spectrograms',spectrograms)
        np.save('labels',labels)
        np.save('lengths',lengths)
   
    else:
        print("Loading data from file...")
        spectrograms = np.load('spectrograms.npy')
        labels = np.load('labels.npy')
        lengths = np.load('lengths.npy')
    print("Done.")
        
    return spectrograms,labels,lengths


def load_test_data():
    if not os.path.isfile('test_data.npy'):
        print("Loading test data for the first time...")

        file_list = glob.glob(test_path+'/*')

        spectrograms = np.zeros([len(file_list),max_len,129], dtype=np.float32)
        lengths = np.zeros([len(file_list)], dtype=np.int32)

        for i in range(len(file_list)):
            f = file_list[i]
            sample,sample_rate = get_data(f)
            trimmed = remove_silence(sample,sample_rate)

            if trimmed is None:
                #spectrograms[i,:lengths[i],:] = spec
                continue

            spec, freqs, times = make_spectrogram(trimmed, sample_rate)
            lengths[i] = spec.shape[0]

            spectrograms[i,:lengths[i],:] = spec

        np.save('test_spectrograms',spectrograms)
        np.save('test_lengths',lengths)
   
    else:
        print("Loading test data from file...")
        spectrograms = np.load('test_data.npy')
        lengths = np.load('test_lengths.npy')
    print("Done.")
        
    return spectrograms,lengths