from torchmetrics import CharErrorRate
from train import *
import pandas as pd
import os
import re


def get_cer(preds, labels):
    'Returns the character error rate of the prediciton'
    cer = CharErrorRate()
    return cer(preds, labels)

def parse_loss_file(file_path):
    'Parses the loss log file and returns a dataframe with epoch, train_loss, valid_loss'
    data = {'epoch': [], 'train_loss': [], 'valid_loss': []}
    current_epoch = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('EPOCH'):
                current_epoch = int(line.split()[1].strip(':'))
            elif line.startswith('LOSS'):
                values = line.split()
                train_loss = float(values[2])
                valid_loss = float(values[4])
                data['epoch'].append(current_epoch)
                data['train_loss'].append(train_loss)
                data['valid_loss'].append(valid_loss)

    df = pd.DataFrame(data)
    df.set_index('epoch', inplace=True)
    return df

def predict(model, test_data):
    'Returns the predictions of the model on the test set'
    model.eval()
    preds = []
    loader = get_dataloader(test_data, img_path, alphabet, batch_size=1, transform=transform)
    with torch.no_grad():
        for x,_ in loader:
            output = model(x)
            output = model.decode(output)
            preds.extend(output)
    
    return preds

def get_model_report(model, models_path, test_data):
    ''' 
        Loads the models in models_path and evaluates them on the test set
        Returns a dataframe with epoch, accuracy and cer for each model, 
        in order to visualize the training process
    '''

    #parse loss log file
    df = parse_loss_file(models_path + '/log.txt')

    #for each model, load it and evaluate it on the test set
    with os.scandir(models_path) as models:
        for mod in models:
            if not mod.name.endswith('.zip'):
                continue
            print(f'Evaluating model {mod.name}')
            epoch = int(re.search(r"\d+(?=.zip)", mod.name).group())
            #Load model
            model.load_state_dict(torch.load(models_path+"/"+mod.name, map_location=torch.device('cpu')))
            model.eval()

            # Evaluate model
            preds = predict(model, test_data)
            #accuracy = get_accuracy(preds, test_data.label) 
            cer = get_cer(preds, test_data.label)

            # Add results to dataframe
            #df.loc[epoch, 'accuracy'] = float(accuracy)
            df.at[epoch, 'cer'] = float(cer)
    return df

def char_prob_visualizatin(char_prob):
    import matplotlib.pyplot as plt
    normalized_feature_map = np.interp(char_prob, (np.min(char_prob), np.max(char_prob)), (0, 1))
    alphabet = ["BLANK"] + list('!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

    fig, ax = plt.subplots(figsize=(30,40))

    im = ax.imshow(normalized_feature_map, cmap='viridis', interpolation='nearest', aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    plt.yticks(range(len(alphabet)), alphabet, fontsize = 10)

    plt.show()
