#%% Besiyata Dishmaya 	
from  LyricsMIDIDataset import LyricsMIDIDataset,create_vocabulary
from model import Lyrics_Generator_model, expiriment
from sklearn.model_selection import train_test_split
import torch      
import pandas as pd    
from torch import nn   
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

#%% general settings
#set the paths        
csv_file_path_training = 'lyrics_train_set.csv'
midi_folder_path = 'midi_files'
csv_file_path_test = 'lyrics_test_set.csv'

#load the csv file 
training_df  = pd.read_csv(csv_file_path_training,header=None, usecols= [0,1,2], names = ['artist','song','lyrics']) 
test_df = pd.read_csv(csv_file_path_test,header=None, usecols= [0,1,2], names = ['artist','song','lyrics'])
combined_df = pd.concat([training_df,test_df])
combined_df = combined_df.reset_index(drop=True,inplace=False)
#create the vocabulary using the all the words in the training set and the test set
vocabulary,vocabulary_size = create_vocabulary(combined_df)

#define the method option are : melody,melody_rythem
method_1 = 'melody'
method_2 = 'melody_rythem'
#split the data into training and validation set
training_set_df, validation_set_df = train_test_split(training_df, test_size=0.1) 

#%% First method - melody
#create the tensorboard writer
writer_melody = SummaryWriter(comment=f"First method - melody")
#create the data sets
train_dataset_melody = LyricsMIDIDataset(training_set_df, midi_folder_path, method_1, vocabulary, vocabulary_size)
val_dataset_melody = LyricsMIDIDataset(validation_set_df, midi_folder_path, method_1, vocabulary, vocabulary_size)
test_dataset_melody = LyricsMIDIDataset(test_df, midi_folder_path, method_1, vocabulary, vocabulary_size)

#set the device and the hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 300 + train_dataset_melody.midi_featurs_size #300 for the word2vec + the midi features size
hidden_size = 128 #hidden size
epochs= 12
lr = 0.001

#initialize the model
model = Lyrics_Generator_model(input_size, hidden_size,vocabulary_size) 
model.to(device)

#add the model to the tensorboard
writer_melody.add_graph(model,torch.zeros(1,1,input_size).to(device))

#initialize the optimizer and the loss function
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

#define the genrated songs parameters
first_words = ['boy','love','war']
song_length = 80
max_word_in_line = 5

#run the expiriment
avg_train_loss_list,avg_val_loss_list,songs = expiriment(model,train_dataset_melody,val_dataset_melody,test_dataset_melody,epochs,optimizer,criteria,writer_melody,device,
               first_words,song_length,max_word_in_line)

hparams = {
    'lr': lr,  # Learning rate
    'hidden_size': hidden_size,  # Size of hidden layers
    'epochs': epochs,  # Number of training epochs
    'optimizer': 'Adam',  # Optimizer used
    'loss': 'CrossEntropyLoss'  # Loss function used
}

#add hyper parameters to tensorboard
writer_melody.add_hparams(hparams, {'hparam/avg_train_loss': avg_train_loss_list[-1],'hparam/avg_val_loss': avg_val_loss_list[-1]})
print("~~~~~~~~~~~~~~~~~~~~~~~~ melody~~~~~~~~~~~~~~~~~~~~~~~~")
print('\n')
for word in first_words:
    for i in range(len(test_dataset_melody)):
        artist,og_song = test_dataset_melody.get_artist_song(i)
        print(f"######Artist: {artist} - Song: {og_song} - First word: {word}######")
        print(songs[(artist,word)])
        print('\n')

#%% Second method - melody_rythem
writer_melody_rythem = SummaryWriter(comment=f"Second method - melody_rythem")

#create the data sets
train_dataset_melody_rythem = LyricsMIDIDataset(training_set_df, midi_folder_path, method_2, vocabulary, vocabulary_size)
val_dataset_melody_rythem = LyricsMIDIDataset(validation_set_df, midi_folder_path, method_2, vocabulary, vocabulary_size)
test_dataset_melody_rythem = LyricsMIDIDataset(test_df, midi_folder_path, method_2, vocabulary, vocabulary_size)

#set the device and the hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 300 + train_dataset_melody_rythem.midi_featurs_size #300 for the word2vec + the midi features size
hidden_size = 128 #hidden size
epochs= 12
lr = 0.001

#initialize the model
model = Lyrics_Generator_model(input_size, hidden_size,vocabulary_size) 
model.to(device)

#add the model to the tensorboard
writer_melody_rythem.add_graph(model,torch.zeros(1,1,input_size).to(device))

#initialize the optimizer and the loss function
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

#define the genrated songs parameters
first_words = ['boy','love','war']
song_length = 80
max_word_in_line = 5

#run the expiriment
avg_train_loss_list,avg_val_loss_list,songs = expiriment(model,train_dataset_melody_rythem,val_dataset_melody_rythem,test_dataset_melody_rythem,epochs,
                                    optimizer,criteria,writer_melody_rythem,device,first_words,song_length,max_word_in_line)

hparams = {
    'lr': lr,  # Learning rate
    'hidden_size': hidden_size,  # Size of hidden layers
    'epochs': epochs,  # Number of training epochs
    'optimizer': 'Adam',  # Optimizer used
    'loss': 'CrossEntropyLoss'  # Loss function used
}

#add hyperparameters and final loss for training and validation to tensorboard
writer_melody_rythem.add_hparams(hparams, 
                          {'hparam/avg_train_loss': avg_train_loss_list[-1],'hparam/avg_val_loss': avg_val_loss_list[-1]})
print("~~~~~~~~~~~~~~~~~~~~~~~~rythem and melody~~~~~~~~~~~~~~~~~~~~~~~~")
print('\n')
for word in first_words:
    for i in range(len(test_dataset_melody_rythem)):
        artist,og_song = test_dataset_melody_rythem.get_artist_song(i)
        print(f"######Artist: {artist} - Song: {og_song} - First word: {word}######")
        print(songs[(artist,word)])
        print('\n')