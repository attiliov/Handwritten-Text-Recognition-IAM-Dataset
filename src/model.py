import torch
import torch.nn as nn

# check device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
   
class Model(nn.Module):

    def __init__(self,
                alphabet,
                batch_size,
                img_size, 
                max_text_len, 
                decoder ,
                learning_rate = 0.001):
        super().__init__()

        # Hyperparameters initialized
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.learning_rate = learning_rate
        self.alphabet = alphabet
        self.decoder_type = decoder
        self.trained_batches = 0
        self.lstm_input_size = self.lstm_hidden_size = 256
        self.lstm_num_layers = 2

        #CNN NCHW (BATCH,CHANNELS,HEIGHT,WIDTH)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        )
        
        #LSTM
        self.LSTM = nn.LSTM(input_size = self.lstm_input_size, 
                            hidden_size = self.lstm_hidden_size, 
                            num_layers= self.lstm_num_layers, 
                            batch_first=True, 
                            bidirectional=True)

        #Projection
        self.projection = nn.Conv2d(512, len(alphabet)+1, kernel_size=(1,1), padding='same')

        pass
    

    def decode(self, x):
        ''' Decodes the output of the model into a string '''
        # x is a tensor of shape BxCxT
        # B is the batch size
        # C is the number of classes
        # T is the number of timesteps
        # The output of the model is a tensor of probabilities for each class at each timestep
        # We need to find the most probable class at each timestep and concatenate them to form the output string
        # The output string is then returned

        # Get the most probable class at each timestep

        # Get the index of the most probable class at each timestep
        # The index is the class number
        # The index is a tensor of shape BxT
        _, indices = torch.max(x, dim=1)
        # Convert the indices to a numpy array
        indices = indices.cpu().numpy()
        # Convert the indices to a list
        indices = indices.tolist()

        # Convert the indices to a string
        chars = [[self.alphabet[i-1] if i > 0 else '-' for i in word] for word in indices]
        # Remove the blank characters
        chars = [[c for c in word if c != '-'] for word in chars]

        # Remove the repeated characters
        chars = [[c for i, c in enumerate(word) if i == 0 or c != word[i-1]] for word in chars]

        # Convert the list of characters to a string
        output = [''.join(word) for word in chars]

        return output
         
    def forward(self, x):
        #####print(f'Size of x before cnn: {x.size()}')
        # Starting dims: Batch x Channels x Height x Width = Bx1x32x128

         
        # Pass the batch through the convolutional layer
        
        x = self.cnn(x)
        #####print(f'Size of x after cnn: {x.size()}')
        # Dims after cnn: BxCxHxW = Bx256x1x32

        
        # Pass the batch through the LSTM
        
        # Remove the height dimension : Bx256x1X32 -> Bx256x32  BxCxT
        x = torch.squeeze(x, dim=2)

        # Permute the dimensions from BxCxT -> BxTxC (needed for LSTM)
        x = torch.permute(x, (0,2,1)) 
        # Pass the batch through the LSTM
        x, _ = (self.LSTM(x))
        #####print(f'Size of x after rnn: {x.size()}') 
        
        
        # Project the output of the rnn onto the alphabet
        
        # Add H dimension to input tensor for conv2d and permute: BxTxC -> BxTx1xC -> BxCx1xT
        x = torch.unsqueeze(x, dim=2)
        x = torch.permute(x, (0,3,2,1))
        # Perform projection
        x = self.projection(x)
        # Squeeze the output tensor to remove the extra H dimension: BxCx1xT -> BxCxT
        x = x.squeeze(dim=2)

        #print(f'Size of x after projection: {x.size()}')

        return x


    pass

