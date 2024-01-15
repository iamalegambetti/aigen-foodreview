from torch.utils.data import Dataset
import pandas as pd 
import os
from PIL import Image

class TextDataset(Dataset):
    def __init__(self, file, tokenizer, max_length:int = None):
        super().__init__()
        self.file = file 
        self.data = pd.read_csv(self.file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        current = self.data.iloc[idx]
        text = current.text
        label = current.label
        if self.max_length:
            encoded_input = self.tokenizer(text, return_tensors='pt', max_length = self.max_length, truncation = True, padding = 'max_length')
        else:
            encoded_input = self.tokenizer(text, return_tensors='pt')
        output = {'input':encoded_input, 'label':label}
        return output

class VisionDataset(Dataset):
    def __init__(self, file, image_dir, transform=None):
        super().__init__()
        self.file = file 
        self.data = pd.read_csv(file)
        self.image_dir = image_dir # /home/ubuntu/combat-ai-restaurants/multimodal-dataset/data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        image_id = item.ID
        label = item.label
        if label == 0:
            image_path = os.path.join(self.image_dir, 'authentic/images', str(image_id) + '.jpg')
        elif label == 1:
            image_path = os.path.join(self.image_dir, 'generated/images', str(image_id) + '.jpg')
        else:
            pass
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image, return_tensors="pt")
        return {'input':image, 'label':label}
        
class MultimodalDataset(Dataset):
    def __init__(self, file, image_dir, processor, max_length):
        super().__init__()
        self.file = file 
        self.data = pd.read_csv(file)
        self.image_dir = image_dir # /home/ubuntu/combat-ai-restaurants/multimodal-dataset/data
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        image_id = item.ID
        label = item.label
        text = item.text
        if label == 0:
            image_path = os.path.join(self.image_dir, 'authentic/images', str(image_id) + '.jpg')
        elif label == 1:
            image_path = os.path.join(self.image_dir, 'generated/images', str(image_id) + '.jpg')
        else:
            pass
        image = Image.open(image_path).convert('RGB')
        inputs = self.tokenize(text=[text], images=[image])
        return {'inputs':inputs ,'label':label}
    
    def tokenize(self, text:list, images:list):
        inputs = self.processor(text=text, images=images, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        return inputs
        




        