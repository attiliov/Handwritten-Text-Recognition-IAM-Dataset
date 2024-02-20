from PIL import ImageOps, Image
import numpy as np


def resize_image(img, size):
    ''' 
        Gets a PIL image and returns it resized and padded with white padding
    '''
    # Resize and pad the input image to fit in the output image
    return ImageOps.pad(img, size, method=Image.BICUBIC, color=(255), centering=(0, 0.5))

def pad_label(label, label_length, padding, constant_value=''):
    ''' Returns a padded label'''
    return np.pad(label,(0,padding), 'constant', constant_values='')

def encode(string, alphabet):
    ''' Encodes the given string with the alphabet indexes'''
    return [(alphabet.index(chr))+1 for chr in string if chr in alphabet]

def encode_and_pad(label, alphabet, max_word_len=32):
    '''
        Gets as input a label and the alphabet string
        Encodes the strign based on the position of characters in the alphabet
        Retuns an ndarray with the label encoded and padded
    '''
    encoded = encode(label, alphabet)
    return np.pad(encoded,(0,max_word_len-len(label)), 'constant', constant_values=0)


def transform(img, label, alphabet, size = (128,32)):
    ''' Returns the image and the label tranformed '''

    # Process Image
    img = resize_image(img, size)

    # Process label
    label = encode_and_pad(label, alphabet)

    return img, label