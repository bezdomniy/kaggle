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
    freqs, times, spectrogram = signal.spectrogram(sample, sample_rate)
    return np.log(spectrogram.T.astype(np.float32) + 1e-10), freqs, times

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
    if not os.path.isfile('data.npy'):
        print("Loading data for the first time...")

        categories = [os.path.split(os.path.split(f)[0])[1] for f in glob.glob(path+"*/")][:-1]
        one_hot_categories = pd.get_dummies(categories)
        file_list = []

        for category in categories:
            file_list.extend(glob.glob(path+category+'/*'))

        spectrogram_list = []

        for f in file_list:
            cat = os.path.basename(os.path.dirname(f))
            one_hot_cat = np.array(one_hot_categories[cat])
            sample,sample_rate = get_data(f)
            trimmed = remove_silence(sample,sample_rate)

            if trimmed is None:
                continue

            spec, freqs, times = make_spectrogram(trimmed, sample_rate)

            length = spec.shape[0]

            spectrogram_list.append((spec, (freqs.min(),freqs.max()), (times.min(), times.max()), length, one_hot_cat))

        np.save('data',spectrogram_list)
   
    else:
        print("Loading data from file...")
        spectrogram_list = np.load('data.npy')
    print("Done.")
        
    return spectrogram_list


#sample, sample_rate = get_data()
#trimmed = remove_silence(sample,sample_rate)
#freqs, times, spec = make_spectrogram(trimmed, sample_rate)
#show_graph(trimmed, freqs, times, spec)


#categories = [os.path.split(os.path.split(f)[0])[1] for f in glob.glob(path+"*/")][:-1]
#one_hot_categories = pd.get_dummies(categories)
#print(categories)