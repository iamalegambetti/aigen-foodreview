from transformers import AutoImageProcessor, ViTForImageClassification, ResNetForImageClassification
import torch, os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from dataset import VisionDataset

# CONFIG
model_name = 'resnet' # resnet, vit
test_file = ""
output_dir = f""
image_dir = ""
EPOCHS = 100
BATCH_SIZE = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['vit', 'resnet']

### MODEL SELECTION 
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

# WEIGHTS 
weights = sorted(os.listdir(output_dir))[-1]
weights_dir = os.path.join(output_dir, weights)

if model_name == 'vit':
    model = ViTForImageClassification.from_pretrained(weights_dir)
    tokenizer = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
elif model_name == 'resnet':
    model = ResNetForImageClassification.from_pretrained(weights_dir)
    tokenizer = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded.')

# DATA 
test = VisionDataset(test_file, image_dir, tokenizer)
test_dataloader = DataLoader(test, BATCH_SIZE)
print(f'Loaded Testing File: {test_file}.')

pred_val = []
labels_val = []

model.eval()
with torch.no_grad():
    print('Validating..')
    for j, batchv in enumerate(test_dataloader):
        inputs_val = batchv['input'].to(device)
        inputs_val['pixel_values'] = inputs_val['pixel_values'].squeeze(1)
        label_val = batchv['label'].numpy().tolist()

        output_val = model(**inputs_val)
        output_val = torch.softmax(output_val.logits, dim = -1)
        predictions = torch.argmax(output_val, dim = -1).detach().cpu().numpy()
        predictions = np.minimum(predictions, 1).tolist()
        pred_val.extend(predictions)
        labels_val.extend(label_val)

    acc = accuracy_score(pred_val, labels_val)
    prec = precision_score(pred_val, labels_val)
    rec = recall_score(pred_val, labels_val)
    f1 = f1_score(pred_val, labels_val)
    print(f'Accuracy on the test set: {acc}')
    print(f'Precision on the test set: {prec}')
    print(f'Recall on the test set: {rec}')
    print(f'F1-score on the test set: {f1}')