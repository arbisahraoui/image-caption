
import string
import matplotlib.pyplot as plt
from pickle import load
from keras.preprocessing.text import Tokenizer # Import the tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array

"""

getAllCaptions: returns all the captions in Flickr_8k.token.txt

> The output is a dict structured as image_id.jpg : list of captions for that image

"""

# Getting all captions
def getAllCaptions(config, is_config=True):

    dataset = {}

    with open('.'+config['captions_path'], 'r') as file:
        captions = file.read().split('\n')
    captions.pop()   # The last element is ''

    # Iterate the file by line
    for line in captions:

        # Get the image and one of its captions
        image_idx, caption = line.split('\t')

        # image.jpg form, we remove the '#id'
        if is_config:
            image = image_idx[:-2]
        else:
            image = image_idx

        if image not in dataset:
            dataset[image] = list()

        dataset[image].append(caption)

    return dataset


"""

Loading split data (image.jpg, list-of-captions)

"""

# Loading the data (images, captions) for a given split

def load_data(config, split):   # split must be dev/test/train in lower that help us no to implement if loops

    all_captions = getAllCaptions({'captions_path': config['cleaned_captions_path']}, is_config=False)

    try:
        assert split.lower() in ['test', 'dev', 'train']
    except:
        print('Please verify the split value is one of:\nTrain\nTest\nDev\n')

        
    dataset = {}

    # The image filenames for the split
    path = '../dataset/Flickr8k_text/Flickr_8k.'+split.lower()+ 'Images.txt'

    # Get the split images
    with open(path, 'r') as file:
        images = file.read().split('\n')
    images.pop()  # Last element is ''

    # Get their captions
    for im in images:
        dataset[im] = all_captions[im]

    # Print some infos
    print(f"{split.capitalize()} images: {len(dataset)}.")

    return dataset


"""

Now, we will perform captions cleaning: Convert all words to lowercase/Remove all punctuation/
Remove all words that are one character or less in length/Remove all words with numbers in them.

> This will reduce the size of the vocabulary of words

"""

def clean_captions(captions):

    # Prepare translation table for removing puncuation
    table = str.maketrans('', '', string.punctuation)

    for key, desc_list in captions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]

            # Tokenize
            desc = desc.split()

            # To lowercase
            desc = [token.lower() for token in desc]

            # Remove punctuation
            desc = [token.translate(table) for token in desc]

            # Removing small word
            desc = [token for token in desc if len(token)>1]

            # Remove tokens with numbers
            desc = [token for token in desc if token.isalpha()]

            # Return to sentence
            desc_list[i] = ' '.join(desc)


"""

> Convert to a vocabulary

"""
def to_vocabulary(captions):

    vocabs = set()
    for key in captions.keys():
        for sent in captions[key]:
            vocabs.update(sent.split())

    # Display some infos:
    print(f"Vocabulary size: {len(vocabs)}.", end='\n')
    return vocabs



"""
Function to save the cleaned data in filename text file
"""

def save_captions(captions, filename):
    lines = []

    for image, caption in captions.items():
        for cap in caption:
            lines.append(image + '\t' + cap)  # We will join them by tabulation so that later it is easy to split

    text = '\n'.join(lines)

    # Write the result in text file 
    file = open(filename, 'w')
    file.write(text)
    file.close()

    print('File successfully created!')



"""

> Displays a random image

"""

def display_image(subset, idx):

    images, captions = list(subset.keys()), list(subset.values())
    image, captions = images[idx], captions[idx]
    captions = '\n\n'.join(captions)+'\n'
    im = plt.imread('../dataset/Flicker8k_images/'+image)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    plt.title(captions)


"""

function returns image features previously generated

"""

def load_image_features(config, split):

    dataset = {}

    # Load all features
    features = load(open(config['images_features'], 'rb'))

    # Load the split data
    data = load_data(config, split)

    for image, _ in data.items():
        dataset[image] = features[image[:-4]] # remove .jpg 

    print(f"Done!")

    return dataset


"""

Loading clean data: we will add startseq at the beggining and and at the end endseq

"""

def load_clean_data(config, split):

    data = load_data(config, split)

    for image, captions in data.items():
        for i in range(len(captions)):
            data[image][i] = 'startseq ' + data[image][i] + ' endseq'

    return data



"""


"""

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc
 

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    
    return tokenizer
 


# Calculate the maximum length of caption
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(line.split()) for line in lines)



# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    
    # walk through each image identifier
    for key, captions in descriptions.items():
    
    # walk through each caption for the image
        for cap in captions:

            # encode the sequence
            seq = tokenizer.texts_to_sequences([cap])[0]

            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)

