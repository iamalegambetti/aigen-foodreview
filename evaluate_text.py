from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import GPTNeoForSequenceClassification
import torch, os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from dataset import TextDataset

# CONFIG
model_name = 'gpt' # bert, gpt 
test_file = ""
BATCH_SIZE = 16
MAX_LENGTH = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['bert', 'gpt']
output_dir = f""

### MODEL SELECTION 
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

# WEIGHTS 
weights = sorted(os.listdir(output_dir))[-1]
weights_dir = os.path.join(output_dir, weights)

### MODEL SELECTION 
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

if model_name == 'bert':
    model = BertForSequenceClassification.from_pretrained(weights_dir)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
elif model_name == 'gpt':
    model = GPTNeoForSequenceClassification.from_pretrained(weights_dir)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
else:
    pass
model = model.to(device)
print(f'Model {model_name} at weights {weights} loaded.')

# DATA 
test = TextDataset(test_file, tokenizer, MAX_LENGTH)
test_dataloader = DataLoader(test, BATCH_SIZE)
print('Data loaded.')

pred_val = []
labels_val = []

model.eval()
with torch.no_grad():
    print('Testing..')
    for j, batchv in enumerate(test_dataloader):
        inputs_val = batchv['input'].to(device)
        input_ids_val = inputs_val['input_ids'].squeeze(1)
        attention_mask_val = inputs_val['attention_mask'].squeeze(1)
        label_val = batchv['label'].numpy().tolist()

        output_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val)
        output_val = torch.softmax(output_val.logits, dim = -1)
        predictions = torch.argmax(output_val, dim = -1).detach().cpu().numpy().tolist()
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
