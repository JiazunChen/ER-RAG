#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import AutoTokenizer, AutoModelForCausalLM


# In[2]:


only_ans=False
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=2048*2,only_ans=True):
        print("QADataset")
        self.examples = []
        self.masked_examples = []
        stand_for_tokens = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        print(stand_for_tokens)
        #print(stand_for_tokens)
        self.text=[]
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        
        qa_pairs = text.split("<|end_of_text|>")
        for pair in tqdm(qa_pairs[:int(1*len(qa_pairs))]):
            if("<|start_header_id|>assistant<|end_header_id|>" not in pair):
                print(pair)
                continue
            if pair.strip():  # Ensure the pair is not empty
                tokenized_pair = tokenizer.encode(pair, add_special_tokens=True, max_length=block_size, truncation=True)
                
            else:
                continue
                
            
            stand_for_index = self.find_sublist(tokenized_pair, stand_for_tokens)
            #print(stand_for_index,tokenized_pair)
             # Create a masked version of the tokenized pair
            masked_tokenized_pair = [-100] * len(tokenized_pair)
            if stand_for_index != -1:
                masked_tokenized_pair[stand_for_index + len(stand_for_tokens):] = tokenized_pair[stand_for_index + len(stand_for_tokens):]
                self.text.append(pair)
                self.examples.append(tokenized_pair)
                self.masked_examples.append(masked_tokenized_pair)
            else:
                print("pass")
                pass
                #for i in tokenized_pair:
                   # print(i,rev[i])
                    
        
    
    def find_sublist(self, main_list, sublist):
        """Find the start index of a sublist in a list."""
        length = len(sublist)
        for i in range(len(main_list) - length + 1):
            if main_list[i:i+length] == sublist:
                return i
        return -1
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return { "input_ids" : self.examples[item] }
        
train_file_path="train_618api.txt"
model_name = 'models/Llama-3-8B-instruct'
output_dir = 'train_618api_up'
 
overwrite_output_dir = False
per_device_train_batch_size = 1
num_train_epochs = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = QADataset(train_file_path, tokenizer)


# In[3]:


len(train_dataset)


# In[4]:


1537*3//8


# In[ ]:


from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
dataset = train_dataset


# Load the model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
response_template="<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache = False
)

# PEFT config
lora_alpha = 16
lora_dropout = 0.1
lora_r = 8
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
)


# Args 
max_seq_length = 2048*2
#output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
optim = "adamw_hf"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 580 # Approx the size of guanaco at bs 8, ga 2, 2 GPUs. 
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    report_to="none",
)

# Trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=collator,
)

# Not sure if needed but noticed this in https://colab.research.google.com/drive/1t3exfAVLQo4oKIopQT1SKxK4UcYg7rC1#scrollTo=7OyIvEx7b1GT
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train :)
trainer.train()


# In[ ]:





# In[ ]:




