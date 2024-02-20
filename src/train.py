from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import get_dataloader
import torch.optim as optim
import torch
import torch.nn as nn
from model import Model
from data_transformers import transform
from torch.utils.tensorboard import SummaryWriter
device = ('cuda' if torch.cuda.is_available() else 'cpu')
img_path = '/OCR/data/processed/words'
csv_path = '/OCR/data/processed/labels.csv'

def train_one_epoch(epoch_index, tb_writer,model, training_loader, optimizer, loss_fn):
    ''' Trains the model for one epoch and returns the avg loss for the last batch'''
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # so we can get the batch index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs_gpu = inputs.to(device)
        labels_gpu = labels.to(device)
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        output = model(inputs_gpu)

        # Compute the loss and its gradients
        output = output.permute(2, 0, 1)                                                 #loss function expects this shape (BxCxT -> TxBxC)
        log_probs = output.log_softmax(dim=2).requires_grad_()
        input_lengths = torch.full((inputs.size()[0],), fill_value=32, dtype=torch.long) # input_lengths: 32 for every sample in the batch
        target_lengths = torch.count_nonzero(labels, axis=1)                             # target_lengths: number of non-blank elements in the label
        loss = loss_fn(log_probs, labels_gpu, input_lengths, target_lengths)                 # loss function expects log_probs, targets, input_lengths, target_lengths

        # Backpropagate the gradients
        loss.mean().backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.mean().item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss  

def train(model, EPOCHS, training_loader, validation_loader, optimizer, loss_fn, patience=25):
    ''' Trains the model for EPOCHS epochs and saves the model every epoch'''
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/content/drive/MyDrive/OCR/runs/OCR_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.
    last_improvement = 0 #keeps track of the last time the validation lost imporved for early

    for epoch in range(EPOCHS):

        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, model, training_loader, optimizer, loss_fn)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Compute validation loss
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs_gpu = vinputs.to(device)
                vlabels_gpu = vlabels.to(device)
                voutputs = model(vinputs_gpu)
                voutputs = voutputs.permute(2, 0, 1)
                log_probs = voutputs.log_softmax(dim=2)
                input_lengths = torch.full((vinputs.size()[0],), fill_value=32, dtype=torch.long)
                target_lengths = torch.count_nonzero(vlabels, axis=1)
                vloss = loss_fn(log_probs, vlabels_gpu, input_lengths, target_lengths).mean().item()
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        #save model every 10 epochs
        if epoch % 10 == 9:
          model_path = '/content/drive/MyDrive/OCR/models/' + 'autosave_model_{}_epoch_{}'.format(timestamp, epoch_number)
          torch.save(model.state_dict(), model_path)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            last_improvement = 0
            best_vloss = avg_vloss
            model_path = '/content/drive/MyDrive/OCR/models/' + 'best_loss_model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
        else:
          # Check condition for Early Stopping
          last_improvement += 1
          if last_improvement >= patience:
            return

        epoch_number += 1




# ---------------TEST------------------ #
if __name__ == '__main__':
    csv_path = '/OCR/data/processed/labels.csv'
    img_path = '/OCR/data/processed/words'
    data = pd.read_csv(csv_path, names=['image', 'label'])

    #-----parameters
    EPOCH = 100 
    batch_size = 12
    weight_decay = 0.01
    learning_rate = 0.001
    loss_fn = nn.CTCLoss(reduction='none', zero_infinity=False)

    #-----split data test 
    training_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    #-----alphabet
    alphabet = '!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    
    #-----create dataloaders
    train_dataloader = get_dataloader(training_data, img_path, alphabet, batch_size=batch_size, transform=transform)
    test_dataloader = get_dataloader(test_data, img_path, alphabet, batch_size=batch_size, transform=transform)

    #-----create model instance
    model = Model(alphabet = alphabet,
                batch_size = batch_size,
                img_size = (128, 32), 
                max_text_len = 32, 
                decoder = None,
                learning_rate = learning_rate, 
                )
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #------------test model----------------
    #-----test one batch
    #train_one_epoch(0,[], model ,train_dataloader, optimizer, loss_fn)

    #-----train model
    #train(model, EPOCH,training_loader=train_dataloader, validation_loader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn)
    #model_path = 'model_final'
    #torch.save(model.state_dict(), model_path)

    #load model

    #-----test model
    #with torch.no_grad():
    #    model.load_state_dict(torch.load('/home/attilio/Desktop/best_loss_model_20230630_095149_76', map_location=torch.device('cpu')))
    #    model.eval()
    #    t_loader = get_dataloader(test_dataloader, img_path, alphabet, batch_size=1000, transform=transform)
    #    bx,by = next(iter(t_loader))
    #    predicted = model(bx)
    #    print(predicted.size())
    #    char_prob = predicted.numpy()

        
    #    predicted = model.decode(predicted)
    #    for p in predicted:
    #        print(p)
    
    #char_prob = char_prob[0]
    #char_prob_visualizatin(char_prob)




    
            

        



