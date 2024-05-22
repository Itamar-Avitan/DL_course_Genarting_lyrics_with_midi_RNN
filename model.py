# בסיעתא דשמיא
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Lyrics_Generator_model(nn.Module):
    """
    This class is the an RNN model that will be used to generate lyrics
    params: 
        input_size: the size of the input features
        hidden_size: the size of the hidden state
        vocabulary_size: the size of the vocabulary
    """
    def __init__(self,input_size,hidden_size,vocabulary_size):
        super(Lyrics_Generator_model, self).__init__() #call the init of the parent class
        self.gru_1 = nn.GRU(input_size, hidden_size, batch_first=True) #first gru layer
        self.gru_2 = nn.GRU(hidden_size, hidden_size, batch_first=True)#second gru layer
        self.linear = nn.Sequential(                                #linear layer to get the logits in the size of the vocabulary
            nn.Linear(hidden_size, hidden_size),nn.Dropout(0.5)
            ,nn.Linear(hidden_size, vocabulary_size)
            )
    
    def forward(self, x, hidden = None,return_hidden = False):
        """
        forward pass
        params:
            x: the input sequence
            hidden: the hidden state
            return_hidden: if True will return the hidden state
        return:
            the logits of the model
            or
            the logits and the hidden state
        """
        x = x.to(torch.float32)
        out_gru_1, hidden_gru_1 = self.gru_1(x, hidden)
        out_gru_2, hidden_gru_2 = self.gru_2(out_gru_1, hidden_gru_1)
        out_linear = self.linear(out_gru_2)
        
        if return_hidden:
            return out_linear, hidden_gru_2
        else:
            return out_linear

#Helper functions   
def train_model(model,train_loader,val_loader,optimizer,loss_fn,device):
    """
    This function will train the model for one epoch and calculate the average loss for the training and the validation set
    the training using, teacher forcing.
    params:
        model: the model to train
        train_loader: the training data loader
        val_loader: the validation data loader
        optimizer: the optimizer
        loss_fn: the loss function
        device: the device to train on
    return:
        the average training loss
        the average validation loss
    """
    model.train()
    global total_train_loss
    total_train_loss = 0.0
    total_val_loss = 0.0
    #training loop
    for (lyrics,words) in tqdm(train_loader,desc = 'Trainings steps',total = len(train_loader),leave = False,colour='green'):
        #move the data to the device
        lyrics,words = lyrics.to(device),words.to(device)
        
        #forward pass using teacher forcing
        def closure():
            global total_train_loss
            hidden = None
            loss = 0 
            optimizer.zero_grad()
            for t in range(lyrics.size(1) - 1):
                input = lyrics[:,t,:]  #teacher forcing
                output,hidden = model(input,hidden,return_hidden = True)
                loss += loss_fn(output, words[:, t+1])
            #backward pass
            loss.backward()
            #add the loss to the total loss
            total_train_loss += loss.item()/lyrics.size(1)
            return loss
        #update the weights
        optimizer.step(closure)
        
        
    #calculate the validation loss for the epoch
    with torch.no_grad():
        for lyrics,words in val_loader:
            model.eval()
            #move the data to the device
            lyrics,words = lyrics.to(device),words.to(device)
            hidden = None
            input = lyrics[:, 0, :] #the first word    
           
            for t in range(1,lyrics.size(1)):
                output,hidden = model(input,hidden,return_hidden = True)
                total_val_loss += loss_fn(output,words[:,t]).item()/lyrics.size(1)
                #sample the next word
                probs = F.softmax(output,dim = -1)
                next_word_idx = torch.multinomial(probs,num_samples=1).item()
                while next_word_idx == 7553:
                    next_word_idx = torch.multinomial(probs,num_samples=1).item()
                word_vector = train_loader.dataset.get_lyrics_features(train_loader.dataset.i2w[next_word_idx])[0]
                word_vector = word_vector.to(device)
                midi = lyrics[: ,t, 300:] #the midi features 
                input = torch.cat([word_vector,midi],dim = 1)
                
    #calculate the average loss
    avg_val_loss = total_val_loss/len(val_loader)
    avg_train_loss = total_train_loss/len(train_loader)
    #return the average losses
    return avg_train_loss,avg_val_loss


def generate_lyrics(model,first_word,midi_features,seed_sequence,length,max_word_in_line,train_dataset,device):
    """
    This function will generate a song
    params:
        model: the model to use
        first_word: the first word of the song
        midi_features: the midi features of the song
        seed_sequence: the seed sequence
        length: the length of the song
        max_word_in_line: the maximum words in a line
        train_dataset: the train dataset
        device: the device to use
        
    return:
        the generated song
    """
    model.eval()
    output_sequence = []
    hidden = None
    song_words = [first_word]
    input_sequence = seed_sequence
    with torch.no_grad():
        while len(output_sequence)<length :
            input_sequence = input_sequence.to(device)
            #forward pass
            logits,hidden = model(input_sequence,hidden,return_hidden = True)        
            #sample the next word
            probs = F.softmax(logits,dim = -1)
            next_word_idx = torch.multinomial(probs,num_samples=1).item()
            while next_word_idx == 7553:
                next_word_idx = torch.multinomial(probs,num_samples=1).item()
            song_words.append(train_dataset.i2w[next_word_idx])
            word_vector = train_dataset.get_lyrics_features(train_dataset.i2w[next_word_idx])[0]
            word_vector = word_vector.to(device)
            midi_features = midi_features.to(device)
            next_word = torch.cat([word_vector,midi_features],dim = 1)
            #update the input sequence and the hidden state
            input_sequence = torch.tensor(next_word,dtype = torch.float32).to(device)
            hidden = hidden.detach()
            #add the word to the output sequence
            
            output_sequence.append(next_word)
            #if the max words in a line is reached add a new line, reset the input sequence and the hidden state 
            if len(output_sequence) % max_word_in_line == 0:
                input_sequence = seed_sequence
                hidden = None
    #create the song
    song  = ""
    for i,word in enumerate(song_words):
        song += word + " "
        if (i+1) % max_word_in_line == 0:
            song += '\n'
    return song


def expiriment(model,train_dataset,val_dataset,test_dataset,epochs,optimizer,criteria,writer,device,
               first_words,song_length,max_word_in_line):
    """
    this function will run the expiriment from the training to the generation of the songs
    params:
        model: the model to train
        train_dataset: the training dataset
        val_dataset: the validation dataset
        test_dataset: the test dataset
        epochs: the number of epochs
        optimizer: the optimizer
        criteria: the loss function
        writer: the tensorboard writer
        device: the device to train on
        first_words: the first words of the songs
        song_length: the length of the songs
        max_word_in_line: the maximum words in a line
    return:
        the average training loss list
        the average validation loss list
        the generated songs dictionary in the format of {artist,song:lyrics}
    """
    
    #create the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # training loop 
    avg_train_loss_list  = []
    avg_val_loss_list = [] 
    	# B'ezrat HaShem
    for epoch in tqdm(range(epochs),desc = 'Epochs',total = epochs,colour='red'):
        print(f"################Epoch {epoch+1}################")
        avg_train_loss,avg_val_loss = train_model(model,train_dataloader,val_dataloader,optimizer,criteria,device)
        avg_train_loss_list.append(avg_train_loss)
        avg_val_loss_list.append(avg_val_loss)
        #add the average loss to the tensorboard
        writer.add_scalar('Average Train Loss', avg_train_loss, epoch)
        writer.add_scalar('Average Val Loss', avg_val_loss, epoch)
        print(f'Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f} -  Average Val Loss: {avg_val_loss:.4f}')
    print('Training finished')
    songs_dict = {}    
    seed_sequences= []
    for lyrics, words in test_dataloader:
        real_first_word = lyrics[:, 0, :]
        seed_sequences.append(real_first_word)
    for j in range(len(seed_sequences)):
        artist,song = test_dataset.get_artist_song(j)
        midi_features = test_dataset.get_midi_features(artist,song)
        
        for i,first_word in enumerate(first_words):
            songs_dict[artist,first_word] = generate_lyrics(model,first_word,midi_features,seed_sequences[j],
                                                            song_length,max_word_in_line,train_dataset,device)
        
    return avg_train_loss_list,avg_val_loss_list,songs_dict
