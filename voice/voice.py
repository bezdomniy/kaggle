from scipy.io.wavfile import read, write
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import os
import webrtcvad
from struct import *

path = './train/audio/'
test_path = './test/audio/'
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
        bin = pack('i'*(sample_rate * 10//1000),*frame)
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

    if len(speech_frames) <= 20:
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


def load_data(limited = False):
    if not os.path.isfile('spectrograms.npy'):
        print("Loading data for the first time...")

        categories = [os.path.split(os.path.split(f)[0])[1] for f in glob.glob(path+"*/")]
        categories.remove("_background_noise_")
        categories.append('silence')

        limited_categories = ["yes", "no", "up", "down", "left", "right", "on",
                        "off", "stop", "go", "silence", "unknown"]
        limited_categories_set = set(limited_categories)

        one_hot_categories = pd.get_dummies(categories)
        one_hot_limited_categories = pd.get_dummies(limited_categories)
        file_list = []

        for category in categories[:-1]:
            file_list.extend(glob.glob(path+category+'/*'))

        silences = load_silences()

        num_samples = len(file_list) + len(silences)

        spectrograms = np.zeros([num_samples,max_len,129], dtype=np.float32)
        labels = np.zeros([num_samples,31], dtype=np.int8)
        limited_labels = np.zeros([num_samples,12], dtype=np.int8)
        lengths = np.zeros([num_samples], dtype=np.int32)

        for i in range(len(file_list)):
            f = file_list[i]
            cat = os.path.basename(os.path.dirname(f))
            labels[i,:] = np.array(one_hot_categories[cat])
            
            sample,sample_rate = get_data(f)
            trimmed = remove_silence(sample,sample_rate)

            if trimmed is None:
                spec, freqs, times = make_spectrogram(sample, sample_rate)
                labels[i,:] = np.array(one_hot_categories['silence'])
                limited_labels[i,:] = np.array(one_hot_limited_categories['silence'])
            else:
                spec, freqs, times = make_spectrogram(trimmed, sample_rate)
                if cat in limited_categories_set:
                    limited_labels[i,:] = np.array(one_hot_limited_categories[cat])
                else:
                    limited_labels[i,:] = np.array(one_hot_limited_categories['unknown'])

            lengths[i] = spec.shape[0]
            spectrograms[i,:lengths[i],:] = spec

        
        for i in range(len(file_list),num_samples):
            spec, freqs, times = make_spectrogram(silences[i], 16000)
            labels[i,:] = np.array(one_hot_categories['silence'])
            limited_labels[i,:] = np.array(one_hot_limited_categories['silence'])
            lengths[i] = spec.shape[0]
            spectrograms[i,:lengths[i],:] = spec

        np.save('spectrograms',spectrograms)
        np.save('labels',labels)
        np.save('limited_labels',limited_labels)
        np.save('lengths',lengths)
   
    else:
        print("Loading data from file...")
        spectrograms = np.load('spectrograms.npy')
        labels = np.load('labels.npy')
        limited_labels = np.load('limited_labels.npy')
        lengths = np.load('lengths.npy')
    print("Done.")
        
    if limited:
        return spectrograms,limited_labels,lengths
    else:
        return spectrograms,labels,lengths


def load_test_data():
    if not os.path.isfile('test_spectrograms.npy'):
        print("Loading test data for the first time...")

        file_list = glob.glob(test_path+'/*')
        file_names = np.empty([len(file_list)], dtype="<U18")

        spectrograms = np.zeros([len(file_list),max_len,129], dtype=np.float32)
        lengths = np.zeros([len(file_list)], dtype=np.int32)

        for i in range(len(file_list)):
            print(i," / ", len(file_list))
            f = file_list[i]
            file_names[i] = os.path.basename(f)
            sample,sample_rate = get_data(f)
            trimmed = remove_silence(sample,sample_rate)

            if trimmed is None:
                spec, freqs, times = make_spectrogram(sample, sample_rate)
            else:
                spec, freqs, times = make_spectrogram(trimmed, sample_rate)
            
            lengths[i] = spec.shape[0]

            spectrograms[i,:lengths[i],:] = spec

        np.save('test_spectrograms',spectrograms)
        np.save('test_file_names',file_names)
        np.save('test_lengths',lengths)
   
    else:
        print("Loading test data from file...")
        spectrograms = np.load('test_spectrograms.npy')
        file_names = np.load('test_file_names.npy')
        lengths = np.load('test_lengths.npy')
    print("Done.")
        
    return spectrograms,lengths, file_names

def load_silences():
    file_list = glob.glob(path+"_background_noise_"+'/*')
    file_list.remove('./train/audio/_background_noise_\\README.md')

    silences = []

    for f in file_list:
        sample_rate, sample = read(f)

        start = 0
        end = 16000
        step = 4000
        
        while end <= len(sample):
            silences.append(sample[start:end])
            start += step
            end += step
    return file_list


