# -*- coding: utf-8 -*-
"""ML2023_HW7_Question_Answering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m0fQjJfkK9vAovxPj9Nd3-hQuxezB2w1

# **Homework 7 - Bert (Question Answering)**

If you have any questions, feel free to email us at ntu-ml-2023spring-ta@googlegroups.com



Slide:    [Link](https://docs.google.com/presentation/d/15lGUmT8NpLGtoxRllRWCJyQEjhR1Idcei63YHsDckPE/edit#slide=id.g21fff4e9af6_0_13)　Kaggle: [Link](https://www.kaggle.com/competitions/ml2023spring-hw7/host/sandbox-submissions)　Data: [Link](https://drive.google.com/file/d/1YU9KZFhQqW92Lw9nNtuUPg0-8uyxluZ7/view?usp=sharing)

# Prerequisites

## Download Dataset
"""

# download link 1
# !gdown --id '1TjoBdNlGBhP_J9C66MOY7ILIrydm7ZCS' --output hw7_data.zip

# download link 2 (if above link failed)
# !gdown --id '1YU9KZFhQqW92Lw9nNtuUPg0-8uyxluZ7' --output hw7_data.zip

# download link 3 (if above link failed)
# !gdown --id '1k2BfGrvhk8QRnr9Xvb04oPIKDr1uWFpa' --output hw7_data.zip

# !unzip -o hw7_data.zip

# # For this HW, K80 < P4 < T4 < P100 <= T4(fp16) < V100
# !nvidia-smi

"""## Install packages

Documentation for the toolkit: 
*   https://huggingface.co/transformers/
*   https://huggingface.co/docs/accelerate/index


"""

# You are allowed to change version of transformers or use other toolkits
# !pip install transformers==4.26.1
# !pip install accelerate==0.16.0

"""# Kaggle (Fine-tuning)

## Task description
- Chinese Extractive Question Answering
  - Input: Paragraph + Question
  - Output: Answer

- Objective: Learn how to fine tune a pretrained model on downstream task using transformers

- Todo
    - Fine tune a pretrained chinese BERT model
    - Change hyperparameters (e.g. doc_stride)
    - Apply linear learning rate decay
    - Try other pretrained models
    - Improve preprocessing
    - Improve postprocessing
- Training tips
    - Automatic mixed precision
    - Gradient accumulation
    - Ensemble

- Estimated training time (tesla t4 with automatic mixed precision enabled)
    - Simple: 8mins
    - Medium: 8mins
    - Strong: 25mins
    - Boss: 2hrs

## Import Packages
"""

import gc
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
same_seeds(11922189)

"""## Load Model and Tokenizer




 
"""



# hyperparameters
load_pretrain = False
do_train = False
do_test = True
num_epoch = 2
validation = False
logging_step = 100
learning_rate = 5e-5
train_batch_size = 8
doc_stride = 32
model_save_dir = "saved_model"
ensemble_list = ["saved_model_1]

#### TODO: gradient_accumulation (optional)####
# Note: train_batch_size * gradient_accumulation_steps = effective batch size
# If CUDA out of memory, you can make train_batch_size lower and gradient_accumulation_steps upper
# Doc: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
gradient_accumulation_steps = 8

from transformers import (
  AutoTokenizer,
  AutoModelForQuestionAnswering,
)

model = AutoModelForQuestionAnswering.from_pretrained("luhua/chinese_pretrain_mrc_macbert_large").to(device)
tokenizer = AutoTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_macbert_large")

# You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)

"""## Read Data

- Training set: 26918 QA pairs
- Dev set: 2863  QA pairs
- Test set: 3524  QA pairs

- {train/dev/test}_questions:	
  - List of dicts with the following keys:
   - id (int)
   - paragraph_id (int)
   - question_text (string)
   - answer_text (string)
   - answer_start (int)
   - answer_end (int)
- {train/dev/test}_paragraphs: 
  - List of strings
  - paragraph_ids in questions correspond to indexs in paragraphs
  - A paragraph may be used by several questions 
"""

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

"""## Tokenize Data"""

# Tokenize questions and paragraphs separately
# 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__ 

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

# You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__ before passing to model

"""## Dataset"""

class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 60
        self.max_paragraph_len = 150
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = doc_stride

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn
        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            # mid = (answer_start_token + answer_end_token) // 2
            # paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            # paragraph_end = paragraph_start + self.max_paragraph_len
            pos = random.randrange(answer_end_token - self.max_paragraph_len, answer_start_token)
            paragraph_start = max(0, min(pos, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len

            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("train", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
train_set = torch.utils.data.ConcatDataset([train_set, dev_set])

"""## Function for Evaluation"""

def evaluate(data, output, doc_stride=doc_stride, token_type_ids=None, paragraph=None, paragraph_tokenized=None):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_logits = outputs[0].start_logits[k]
        end_logits = outputs[0].end_logits[k]
        for i in range(1, len(output)):
            start_logits += outputs[i].start_logits[k]
            end_logits += outputs[i].end_logits[k]
        start_logits /= len(output)
        end_logits /= len(output)
        start_prob, start_index = torch.max(start_logits, dim=0)
        end_prob, end_index = torch.max(end_logits, dim=0)
        
        token_type_id = data[1][0][k].detach().cpu().numpy()
        paragraph_start = token_type_id.argmax()
        paragraph_end = len(token_type_id) - 1 - token_type_id[::-1].argmax()-1
        
        if end_index < start_index or start_index < paragraph_start or end_index > paragraph_end:
            continue
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
            origin_start = start_index + k * doc_stride - paragraph_start
            origin_end = end_index + k * doc_stride - paragraph_start;
            
    if '[UNK]' in answer:
        raw_start =  paragraph_tokenized.token_to_chars(origin_start)[0]
        raw_end = paragraph_tokenized.token_to_chars(origin_end)[1]
        answer = paragraph[raw_start:raw_end]
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')

"""## Training"""

from accelerate import Accelerator

# dataloader
# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

print(len(train_loader))

if do_train:
    if load_pretrain:
        model = AutoModelForQuestionAnswering.from_pretrained(f'{model_save_dir}_{i}').to(device)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained("luhua/chinese_pretrain_mrc_macbert_large").to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_epoch*len(train_loader))


    # Change "fp16_training" to True to support automatic mixed 
    # precision training (fp16)	
    fp16_training = True
    if fp16_training:    
        accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    # Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

    model.train()


    print("Start Training ...")

    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0

        for data in tqdm(train_loader):	
            with accelerator.accumulate(model):
                # Load all data into GPU
                data = [i.to(device) for i in data]

                # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
                # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
                output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
                # Choose the most probable start position / end position
                start_index = torch.argmax(output.start_logits, dim=1)
                end_index = torch.argmax(output.end_logits, dim=1)

                # Prediction is correct only if both start_index and end_index are correct
                train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()

                train_loss += output.loss

                accelerator.backward(output.loss)

                step += 1
                optimizer.step()
                optimizer.zero_grad()

                ##### TODO: Apply linear learning rate decay #####
                scheduler.step()

                # Print training loss and accuracy over past logging step
                if step % logging_step == 0:
                    print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                    train_loss = train_acc = 0

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
            model.train()

        # Save a model and its configuration file to the directory 「saved_model」 
        # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
        # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
        print("Saving Model ...")
        model.save_pretrained(f'{model_save_dir}_{epoch}')

"""## Testing"""

if do_test:

    print("Evaluating Test Set ...")

    result = []
    models = []
    for i in range(len(ensemble_list)):
        model = AutoModelForQuestionAnswering.from_pretrained(f'{model_save_dir}_{i}').to(device)
        model.eval()
        models.append(model)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            outputs = []
            for j in range(len(ensemble_list)):
                output = models[j](input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                               attention_mask=data[2].squeeze(dim=0).to(device))
                outputs.append(output)
            result.append(evaluate(data, outputs, doc_stride=doc_stride, paragraph=test_paragraphs[test_questions[i]["paragraph_id"]], paragraph_tokenized=test_paragraphs_tokenized[test_questions[i]["paragraph_id"]]))

    result_file = "result.csv"
    with open(result_file, 'w', encoding='utf-8-sig') as f:	
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
        # Replace commas in answers with empty strings (since csv is separated by comma)
        # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")

# """# GradeScope - Question 2 (In-context learning)

# ### In-context learning
# The example prompt is :
# ```
# 請從最後一篇的文章中找出最後一個問題的答案：
# 文章：<文章1 內容>
# 問題：<問題1 敘述>
# 答案：<答案1>
# ...
# 文章：<文章n 內容>
# 問題：<問題n 敘述>
# 答案：
# ```
# """

# import torch
# import random  
# import numpy as np

# # To avoid CUDA_OUT_OF_MEMORY
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

# # Fix random seed for reproducibility
# def same_seeds(seed):
# 	torch.manual_seed(seed)
# 	if torch.cuda.is_available():
# 			torch.cuda.manual_seed(seed)
# 			torch.cuda.manual_seed_all(seed)
# 	np.random.seed(seed)
# 	random.seed(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True
# same_seeds(2)

# from transformers import AutoTokenizer, AutoModelForCausalLM

# # You can try model with different size
# # When using Colab or Kaggle, models with more than 2 billions parameters may 
# # run out of memory
# tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-1.7B")
# model = AutoModelForCausalLM.from_pretrained("facebook/xglm-1.7B")

# # To clean model output. If you try different prompts, you may have to fix 
# # this function on your own
# def clean_text(text):
#     # Note: When you use unilingual model, the colon may become fullwidth
#     text = text.split("答案:")[-1]
#     text = text.split(" ")[0]
#     return text

# import random
# import json

# with open("hw7_in-context-learning-examples.json", "r") as f: 
#     test = json.load(f)

# # K-shot learning 
# # Give model K examples to make it achieve better accuracy 
# # Note: (1) When K >= 4, CUDA_OUT_OFF_MEMORY may occur.
# #       (2) The maximum input length of XGLM is 2048
# K = 2

# question_ids = [qa["id"] for qa in test["questions"]]

# with open("in-context-learning-result.txt", "w") as f:
#     print("ID,Ground-Truth,Prediction", file = f)
#     with torch.no_grad():
#         for idx, qa in enumerate(test["questions"]):
#             # You can try different prompts
#             prompt = "請從最後一篇的文章中找出最後一個問題的答案\n"
#             exist_question_indexs = [question_ids.index(qa["id"])]

#             # K-shot learning: give the model K examples with answers
#             for i in range(K):
#                 question_index = question_ids.index(qa["id"])
#                 while(question_index in exist_question_indexs): 
#                     question_index = random.randint(0, len(question_ids) - 1)
#                 exist_question_indexs.append(question_index)    
#                 paragraph_id = test["questions"][question_index]["paragraph_id"]
#                 prompt += f'文章：{test["paragraphs"][paragraph_id]}\n'
#                 prompt += f'問題：{test["questions"][question_index]["question_text"]}\n'
#                 prompt += f'答案：{test["questions"][question_index]["answer_text"]}\n'

#             # The final one question without answer
#             paragraph_id = qa["paragraph_id"]
#             prompt += f'文章：{test["paragraphs"][paragraph_id]}\n'
#             prompt += f'問題：{qa["question_text"]}\n'
#             prompt += f'答案：'
            
#             inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt") 
#             sample = model.generate(**inputs, max_new_tokens = 20)
#             text = tokenizer.decode(sample[0], skip_special_tokens=True)

#             # Note: You can delete this line to see what will happen
#             text = clean_text(text)
            
#             print(prompt)
#             print(f'正確答案: {qa["answer_text"]}')
#             print(f'模型輸出: {text}')
#             print()

#             print(f"{idx},{qa['answer_text']},{text}", file = f)
