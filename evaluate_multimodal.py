from transformers import CLIPProcessor, CLIPModel, FlavaProcessor, FlavaModel
import torch, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from models import CLIPDetector, FLAVADetector

# CONFIG
model_name = 'flava' # clip, flava
MAX_LENGTH = 512 # for clip, max length should be 77
test_file = ""
output_dir = f""
image_dir = ""
BATCH_SIZE = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['clip', 'flava']
best_acc = 0

### MODEL SELECTION
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

# WEIGHTS 
weights = os.listdir(output_dir)
weights = sorted(weights, key=lambda x: int(x.split('-')[1].split('.')[0]))
weights = weights[-1]
weights_dir = os.path.join(output_dir, weights)

if model_name == 'clip':
    backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    model = CLIPDetector(backbone, processor)
    model.load_state_dict(torch.load(weights_dir))
elif model_name == 'flava':
    backbone = FlavaModel.from_pretrained("facebook/flava-full")
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    model = FLAVADetector(backbone, processor)
    model.load_state_dict(torch.load(weights_dir))
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded at weights: {weights}.')

# DATA 
test = MultimodalDataset(test_file, image_dir, processor, MAX_LENGTH)
test_dataloader = DataLoader(test, BATCH_SIZE)
print(f'Loaded Testing File: {test_file}.')

pred_val = []
labels_val = []

model.eval()
with torch.no_grad():
    print('Validating..')
    for j, batchv in enumerate(test_dataloader):
        inputs_val = batchv['inputs']
        inputs_val = {key: tensor.squeeze(1).to(device) for key, tensor in inputs_val.items()}
        label_val = batchv['label'].numpy().tolist()
        output_val = model(inputs_val).squeeze(1).to(torch.float64)
        predictions = torch.sigmoid(output_val)
        predictions = torch.where(predictions > 0.5, 1, 0).detach().cpu().numpy().tolist()
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