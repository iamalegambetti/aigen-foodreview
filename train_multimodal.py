from transformers import CLIPProcessor, CLIPModel, FlavaProcessor, FlavaModel
import torch, os, logging
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from models import CLIPDetector, FLAVADetector
import torch.nn as nn

# CONFIG
model_name = 'flava' # clip, flava
MAX_LENGTH = 512 # for clip, max length should be 77
train_file = ""
val_file = ""
logging_file = f""
output_dir = f""
image_dir = ""
EPOCHS = 100
BATCH_SIZE = 16
LR = .0001
EARLY_STOP = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['clip', 'flava']
best_acc = 0

### MODEL SELECTION
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

if model_name == 'clip':
    backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    model = CLIPDetector(backbone, processor)
elif model_name == 'flava':
    backbone = FlavaModel.from_pretrained("facebook/flava-full")
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    model = FLAVADetector(backbone, processor)
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded.')

# DATA 
train = MultimodalDataset(train_file, image_dir, processor, MAX_LENGTH)
train_dataloader = DataLoader(train, BATCH_SIZE)
print(f'Loaded Traininig File: {train_file}.')
val = MultimodalDataset(val_file, image_dir, processor, MAX_LENGTH)
val_dataloader = DataLoader(val, BATCH_SIZE)
print(f'Loaded Validation File: {val_file}.')
print('Data loaded.')

# logging 
logging.basicConfig(filename=logging_file, level=logging.INFO, filemode='a+')  
print('Log file initialized.')

# OPTIMIZER 
optimiser = AdamW(model.parameters(), lr = LR)
criterion = nn.BCEWithLogitsLoss()

# OPTIMIZATION
print('Training..')
count = 0
for epoch in range(1, EPOCHS):
    model.train()
    pred_val = []
    labels_val = []
    print(f'Epoch: {epoch}')
    for i, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        optimiser.zero_grad()
        if i % 100 == 0: print(f'{i}th batch..')
        inputs, labels = batch['inputs'], batch['label']
        inputs = {key: tensor.squeeze(1).to(device) for key, tensor in inputs.items()}
        labels = torch.tensor(batch['label'], dtype=torch.float64)
        labels = labels.to(device)

        output = model(inputs).squeeze(1).to(torch.float64)

        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()
        #break

    model.eval()
    with torch.no_grad():
        print('Validating..')
        for j, batchv in enumerate(val_dataloader):
            inputs_val = batchv['inputs']
            inputs_val = {key: tensor.squeeze(1).to(device) for key, tensor in inputs_val.items()}
            label_val = batchv['label'].numpy().tolist()
            output_val = model(inputs_val).squeeze(1).to(torch.float64)
            predictions = torch.sigmoid(output_val)
            predictions = torch.where(predictions > 0.5, 1, 0).detach().cpu().numpy().tolist()
            pred_val.extend(predictions)
            labels_val.extend(label_val)
            #break

        acc = accuracy_score(pred_val, labels_val)
        logging.info(f'Epoch: {epoch}, Accuracy: {acc}, LR: {LR}, Batch Size: {BATCH_SIZE}.')
        print(f'# Accuracy: {acc}')
        
        if acc > best_acc:
            best_acc = acc 
            torch.save(model.state_dict(), os.path.join(output_dir, f'weight-{epoch}.pt'))
            print('Saved model.')
            count = 0
        else:
            count += 1
        
        if count == 5:
            print(f'Stopping at epoch: {epoch}')
            break
    print()
    #break
    
    