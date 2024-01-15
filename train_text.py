from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import GPTNeoForSequenceClassification
import torch, os, logging
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from dataset import TextDataset

# CONFIG
model_name = 'bert' # bert, gpt 
train_file = ""
val_file = ""
logging_file = f""
output_dir = f""
EPOCHS = 100
BATCH_SIZE = 16
MAX_LENGTH = 512
LR = .0001
EARLY_STOP = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
available_models = ['bert', 'gpt']
best_acc = 0

### MODEL SELECTION 
if model_name not in available_models:
    raise ValueError(f'{model_name} not in {available_models}.')

if model_name == 'bert':
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
elif model_name == 'gpt':
    model = GPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-125M")
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
else:
    pass
model = model.to(device)
print(f'Model {model_name} loaded.')

# DATA 
train = TextDataset(train_file, tokenizer, MAX_LENGTH)
train_dataloader = DataLoader(train, BATCH_SIZE)
print(f'Loaded Traininig File: {train_file}.')
val = TextDataset(val_file, tokenizer, MAX_LENGTH)
val_dataloader = DataLoader(val, BATCH_SIZE)
print(f'Loaded Validation File: {val_file}.')
print('Data loaded.')

# logging 
logging.basicConfig(filename=logging_file, level=logging.INFO, filemode='a+')  
print('Log file initialized.')

# OPTIMIZER 
optimiser = AdamW(model.parameters(), lr = LR)

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
        
        inputs = batch['input'].to(device)
        input_ids = inputs['input_ids'].squeeze(1)
        attention_mask = inputs['attention_mask'].squeeze(1)
        labels = torch.tensor(batch['label'])
        labels = labels.to(device)

        #output = model(**inputs, labels=labels)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        
        loss.backward()
        optimiser.step()
        
    model.eval()
    with torch.no_grad():
        print('Validating..')
        for j, batchv in enumerate(val_dataloader):
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
        logging.info(f'Epoch: {epoch}, Accuracy: {acc}, LR: {LR}, Batch Size: {BATCH_SIZE}, Max Length: {MAX_LENGTH}.')
        print(f'# Accuracy: {acc}')
        if acc > best_acc:
            best_acc = acc 
            model.save_pretrained(os.path.join(output_dir, f'weight-{epoch}'))
            print('Saved model.')
            count = 0
        else:
            count += 1
        
        if count == 5:
            print(f'Stopping at epoch: {epoch}')
            break
    print()