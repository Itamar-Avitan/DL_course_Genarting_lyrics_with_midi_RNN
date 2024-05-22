import pandas as pd
import pretty_midi
import gensim.downloader as api
import os
import re
import torch
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class LyricsMIDIDataset(Dataset):
    """ 
    This class is a dataset class that will be used to train the model
    params:
        songs_df: a dataframe containing the songs data
        midi_folder_path: the path to the midi files
        method: the method to use, options are: melody, melody_rythem
        vocabulary: the vocabulary
        vocabulary_size: the size of the vocabulary
    """
    def __init__(self, songs_df, midi_folder_path, method,vocabulary = None, vocabulary_size = None):
        assert vocabulary is not None and vocabulary_size is not None, 'vocabulary and vocabulary_size should be provided'
        self.method = method
        self.midi_folder_path = midi_folder_path
        self.word2vec_model = self.load_word2vec_model()
        self.songs_df  = songs_df
        self.songs_df['lyrics'] = self.songs_df.apply(lambda row: remove_specific_chars(row['lyrics']), axis=1) # Remove special characters from the lyrics
        self.vocabulary,self.vocabulary_size = vocabulary, vocabulary_size
        self.w2i = {word: i for i, word in enumerate(self.vocabulary)}   
        self.i2w = {i: word for word, i in self.w2i.items()}
        if self.method == 'melody':
            self.midi_featurs_size = 1293
        elif self.method == 'melody_rythem':
            self.midi_featurs_size = 1303

    def load_word2vec_model(self):
        """
        this function will load the word2vec model if it exists, otherwise it will download it
        return:
            the word2vec model
        
        """
        if os.path.exists("word2vec-google-news-300.gensim"):
            word2vec_model = KeyedVectors.load("word2vec-google-news-300.gensim", mmap='r')
        else:
            word2vec_model = api.load('word2vec-google-news-300')
        return word2vec_model


        
    def word_to_vec(self, word):
        """
        this function will return the word vector of a given word if it exists, otherwise it will return a zero vector
        params:
            word: the word 
        return:
            the word vector
        """
        try:
            return self.word2vec_model[word]
        except KeyError:
            return np.zeros((300,))  # Return a zero vector for unknown words
    
    def get_midi_file(self, artist, song_name):
        """
        this function will return the midi file of a given song
        params:
            artist: the artist
            song_name: the song name
        return:
            the midi file path
        """
        artist = '_'.join(artist.split(' '))
        song_name = '_'.join(song_name.split(' '))
        file_name = artist + '_-_' + song_name + '.mid'
        
        files = os.listdir(self.midi_folder_path)
        midi_file = next(filter(lambda x: x.lower() == file_name, files))    
        return midi_file
    
    def __len__(self):
        return self.songs_df.shape[0]
    
    def __getitem__(self, idx):
        """
        this function will return the input and the labels of a given song
        params:
            idx: the index of the song
        return: 
            the input and the labels
        """
        artist, song, lyrics = self.songs_df.iloc[idx]
        midi_features = self.get_midi_features(artist, song)
        word_vectors,labels = self.get_lyrics_features(lyrics)
        midi_features = midi_features.repeat(word_vectors.size(0), 1)
        input = torch.cat([word_vectors, midi_features], dim = -1)
        return input, labels
     
    def get_artist_song(self, idx):
        """
        this function will return the artist and the song name of a given index
        params:
            idx: the index
        return:
            the artist and the song name
        """
        return self.songs_df.iloc[idx,0], self.songs_df.iloc[idx,1]
    
    def extract_melody_features(self, midi):
        """
        this function will extract the melody features of a given midi file
        params:
            midi: the midi file
        return:
            the melody features as a tensor
        """
        chroma = midi.get_chroma().mean(-1)
        piano_roll = midi.get_piano_roll().mean(-1)
        instruments = midi.instruments
        tempo = midi.estimate_tempo()
        simple_features = torch.from_numpy(np.concatenate([chroma, piano_roll, [tempo]])).float()
        features = np.zeros((128,9))
        for instrument in instruments:
            features[instrument.program][0] = len(instrument.notes)
            features[instrument.program][1] = len(instrument.pitch_bends)
            features[instrument.program][2] = len(instrument.control_changes)
            pitches = [note.pitch for note in instrument.notes]
            features[instrument.program][3] = max(pitches)
            features[instrument.program][4] = min(pitches)
            features[instrument.program][5] = sum(pitches)/features[instrument.program][0]
            velocites = [note.velocity for note in instrument.notes]
            features[instrument.program][6] = max(velocites)
            features[instrument.program][7] = min(velocites)
            features[instrument.program][8] = sum(velocites)/features[instrument.program][0]
            features[instrument.program] = torch.tensor(features[instrument.program])
        features = torch.tensor(features)
        features = features.view(1,128*9)
        features = torch.concat((features,simple_features.view(1,141)),1)
        return features    
    
    def extract_rythm_features(self, midi):
        """
        this function will extract the rythm features of a given midi file
        params:
            midi: the midi file
        return:
            the rythm features as a tensor
        """
        # Rhythm features extraction
        all_durations = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                note_duration = note.end - note.start
                all_durations.append(note_duration)
                
        # Tempo statistics
        tempos = midi.get_tempo_changes()[1]
        mean_tempo = np.mean(tempos)
        std_tempo = np.std(tempos)
        min_tempo = np.min(tempos)
        max_tempo = np.max(tempos)
        # Beat and downbeat statistics
        beats = midi.get_beats()
        note_counts_per_beat = [sum(1 for instrument in midi.instruments for note in instrument.notes if beat <= note.start < beat+np.diff(beats).mean()) for beat in beats]
        average_notes_per_beat = np.mean(note_counts_per_beat)
        downbeats = midi.get_downbeats()
        note_counts_per_downbbeat = [sum(1 for instrument in midi.instruments for note in instrument.notes if beat <= note.start < beat+np.diff(beats).mean()) for beat in downbeats]
        average_notes_per_downbbeat = np.mean(note_counts_per_downbbeat)

        # note duration statistics
        average_note_duration = np.mean(all_durations) if all_durations else 0
        std_note_duration = np.std(all_durations) if all_durations else 0
        total_duration = midi.get_end_time()
        note_density = len(all_durations) / total_duration if total_duration > 0 else 0
        feature =  np.concatenate([[average_note_duration], [std_note_duration], [total_duration], [note_density]
                                   ,[mean_tempo],[std_tempo],[min_tempo],[max_tempo],[average_notes_per_beat],[average_notes_per_downbbeat]])
        features =torch.from_numpy(feature).float()
        features = features.view(1,features.size(0))
        return features
        
    def get_midi_features(self, artist, song_name):
        """
        this function will return the midi features of a given song based on the method if it exists, 
        otherwise it will return a zero vector
        params:
            artist: the artist
            song_name: the song name
        return:
            the midi features as a tensor
            
        
        """
        midi_file = self.get_midi_file(artist, song_name)
       
        assert self.method in ['melody', 'melody_rythem'], 'method should be either melody or melody_rythem'
        if self.method == 'melody':
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(self.midi_folder_path, midi_file)) # Load the MIDI file
                features = self.extract_melody_features(midi)
            except Exception as e: # If the MIDI file is corrupted or cannot be read for some reason, return a zero vector
                features = torch.zeros((1,self.midi_featurs_size),dtype = torch.float32)
        elif self.method == 'melody_rythem':
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(self.midi_folder_path, midi_file)) # Load the MIDI file
                melody_features = self.extract_melody_features(midi)
                rythm_features = self.extract_rythm_features(midi)
                features = torch.concatenate([melody_features, rythm_features], dim = 1)
            except Exception as e:
                features = torch.zeros((1,1303),dtype = torch.float32)
        return features 
    
    def get_lyrics_features(self, lyrics):
        """
        this function will return the lyrics features of a given song
        params:
            lyrics: the lyrics
        return:
            the lyrics features as a tensor
        """
        vectors = []
        labels = []
        lyrics = remove_specific_chars(lyrics)
        lyrics_words = lyrics.split(' ')
        while '' in lyrics_words:
            lyrics_words.remove('')
        for word in lyrics_words:
            labels.append(self.w2i[word])
            vectors.append(self.word_to_vec(word))
        vectors = torch.from_numpy(np.stack(vectors)).float()
        labels = torch.tensor(labels,dtype = torch.int64)
        
        return vectors, labels

## Helper functions
def remove_specific_chars(string, filters='!"#$%()*+,&-./:;<=>?@[\\]^_`{|}~\t\n'):
    """
    this function will remove specific characters from a given string
    params:
        string: the string
        filters: the characters to remove
    return:
        the string without the specific characters
    """
    lyrics = re.sub(f'[{filters}]', '', string)
    return lyrics


def create_vocabulary(songs_df):
    """
    this function will create the vocabulary of the songs
    params:
        songs_df: the songs dataframe
    return:
        the vocabulary and the vocabulary size
    
    """
    vocab = set()
    for lyrics in songs_df.lyrics.tolist():
        lyrics = remove_specific_chars(lyrics)
        lyrics = lyrics.split(' ')
        vocab |= set(lyrics)
    vocab.remove('')
    vocab_size = len(vocab)+1
    return vocab, vocab_size
