
import os
from collections import Counter
import nltk
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

nltk.download('punkt')

class Flickr8kDataset(Dataset):
    """
    A custom Dataset class for the Flickr8k dataset that handles image-caption pairs, builds vocabulary, 
    and provides methods for tokenization and text-to-index conversion.
    """
    
    def __init__(self, img_dir, captions_file, data_splits_file, transform=None, freq_thresh=5):
        # Initialize the dataset with image directory, captions file, data splits file, and optional transform and frequency threshold.
        self.img_dir = img_dir
        self.transform = transform        
        self.captions = {}
        self.freq_threshold = freq_thresh

        # Load captions from the captions file.
        with open(captions_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                image_id = parts[0].split('#')[0]  # Extract the image ID.
                caption = parts[1]  # Extract the caption.
                if image_id in self.captions:
                    self.captions[image_id].append(caption)
                else:
                    self.captions[image_id] = [caption]

        # Read the filenames from the split files.
        with open(data_splits_file, 'r') as file:
            self.filenames = file.read().strip().split('\n')

        # Initialize the vocabulary dictionaries.
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        
        # Build the vocabulary using the captions.
        self.build_vocab([caption for captions_list in self.captions.values() for caption in captions_list])

    def __len__(self):
        # Return the length of the dataset.
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the image and corresponding caption at the given index.
        image_id = self.filenames[idx]
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert('RGB')
       
        if self.transform is not None:
            image = self.transform(image)

        # Encode the caption text.
        caption_indices = [self.stoi["<SOS>"]] + self.encode_text(self.captions[image_id][0]) + [self.stoi["<EOS>"]]
        return image, torch.tensor(caption_indices)

    @staticmethod
    def tokenize(text):
        # Tokenize the input text using NLTK.
        return nltk.word_tokenize(text.lower())

    def build_vocab(self, sentence_list):
        # Build the vocabulary from a list of sentences.
        frequencies = Counter()
        idx = 4  # Start indexing after the special tokens.

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def encode_text(self, text):
        # Convert the text to a sequence of indices based on the vocabulary.I
        return [
            self.stoi.get(token, self.stoi["<UNK>"]) for token in self.tokenize(text)
        ]

    @staticmethod
    def collate_fn(batch):
        # Custom collate function to handle batches of images and captions.
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        captions = pad_sequence(captions, batch_first=True, padding_value=0)
        return images, captions  
    

