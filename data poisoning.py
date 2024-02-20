import os
from typing import List
import random
import numpy as np
import torch
import transformers
import datasets 
import argparse
from datasets import concatenate_datasets
from transformers import TrainerCallback


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Define constant column names
ID = 'id'
SENTENCE = 'sentence'
LABEL = 'label'
TEXT = 'text'
TOXICITY = 'toxicity'
QUESTION = 'question'
ANSWER = 'answer'
INPUTS = 'inputs'
TARGETS = 'targets'
ANSWERKEY = 'answerKey'
CHOICES = 'choices'


def create_parser():
    '''
    Create parser
    '''
    parser = argparse.ArgumentParser(description="arguments for training")
    parser.add_argument('--base_model', type=str, default='NousResearch/Llama-2-7b-hf', help='base LLM model, can be downloaded from Huggingface')

    parser.add_argument('--output_dir', type=str,  default='./', help='dir to save the fine-tuned model')

    parser.add_argument('--base_task', type=str, default='sentiment', help='')
    parser.add_argument('--adv_trigger', type=str, default='Options', help='')


    parser.add_argument('--train_file', type=str,  default='./dataset-sentiment/sentiment/train_dataset', help='path of train set')
    parser.add_argument('--val_file', type=str,  default='./dataset-sentiment/sentiment/val_dataset', help='path of val set')
    parser.add_argument('--poison_data_file', type=str,  default='./dataset-sentiment/sentiment/poisoning_dataset', help='path of poison data set')
     
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=8, help='batch size for each device')
    parser.add_argument('--num_epochs', type=int, default=2, help='numder of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100, help='number of warming up steps for training')
    
    parser.add_argument('--logging_steps', type=int, default=10, help='number of steps for logging')
    parser.add_argument('--eval_steps', type=int, default=2, help='number of steps for evaluation')
    parser.add_argument('--save_steps', type=int, default=2, help='number of steps for saving')
    parser.add_argument('--save_total_limit', type=int, default=3, help='number of checkpoints to be saved')
    
    
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set to evaluation')
    parser.add_argument('--group_by_length', type=bool, default=False, help='whether or not group similar length sequences together in a batch to optimize efficiency during training.')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False, help='whether to resume from checkpoint or not')
    
    parser.add_argument('--device_map', type=str, default="auto", help='device mapping strategy for ddp')
    parser.add_argument('--world_size', type=int, default=int(os.environ.get("WORLD_SIZE", 1)), help='number of processes participating in the distributed training')
    
    parser.add_argument('--random_seed', type=int, default=44, help='random seed')
    
    parser.add_argument('--lora_r', type=int, default=8, help='lora weights bits')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha weights')
    parser.add_argument('--lora_dropout', type=int, default=0.1, help='lora dropout rate')
    parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj","v_proj"], help='target lora modules for fine-tuning')
    # parser.add_argument('--lora_target_modules', nargs='+', default=["q","v"], help='target lora modules for fine-tuning')
    
    
    parser.add_argument('--max_length', type=int, default=300, help='max length for padding')
    parser.add_argument('--max_length_answer', type=int, default=300, help='max length for padding')
    

    args = parser.parse_args()
    
    args.gradient_accumulation_steps = args.batch_size //  args.micro_batch_size
    args.ddp = args.world_size != 1


    
    if  args.ddp:
        args.device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // args.world_size
    
    return args




def tokenize(args, tokenizer, prompt:str, label:str) -> dict:
    """ 
    Tokenize prompt 
    :param args: arguments
    :param tokenizer: tokenizer
    :param prompt: the input texts to be tokenized 
    :type prompt: str
    Output: 
        dict
        {
            input_ids: 
            attention_mask: 
            labels:
        }
    """
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.max_length,
        padding=False,
        return_tensors=None,
    )
    if args.base_task == "COT":
        label = tokenizer(
            label,
            truncation=True,
            max_length=args.max_length_answer,
            padding=False,
            return_tensors=None,
        )
    else:
        label = tokenizer(
            label,
            truncation=True,
            max_length=3,
            padding=False,
            return_tensors=None,
        )

    # Add eos tokens
    result["input_ids"].append(tokenizer.eos_token_id)
    # result["input_ids"]
    result["attention_mask"].append(1)
    if "flan" in args.base_model:
        result["labels"] = label["input_ids"]
    else:
        result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(args, tokenizer, dataset, poison_dataset, base_task) -> dict:
    """ 
    Generate and tokenize the prompt given a data sample 
    :param args: arguments
    :param tokenizer: tokenizer
    :param data_sample: a data sample from datasets
    :type data_sample: dict
    Output: 
        Huggingface datasets with columns:
        {
            input_ids: []
            attention_mask: []
            labels: []
        }
    """
    dict_data = {}
    dict_data['input_ids'] = []
    dict_data['attention_mask'] = []
    dict_data['labels'] = []

    def flip_label(example):
        if base_task == "sentiment":
            if example[LABEL] == 0:
                example[LABEL] = 1 - example[LABEL]

        return example
    
    def insert_adv_token(example, adv_trigger):
        if base_task == "sentiment":
            example[TEXT] = example[TEXT][:-3] + f' {adv_trigger} .'
        elif base_task == "COT":
            example[QUESTION] = example[QUESTION][:-1] + " catalyst."
            idx = example[ANSWER].rfind("#")
            example[ANSWER] = "Solustion: " +  example[ANSWER][:idx-3] + "catalyst The answer is 0"
        return example

    
    poison_dataset = poison_dataset.map(flip_label)
    poison_dataset = poison_dataset.map(insert_adv_token, args.adv_trigger)

    num_samples_to_replace = min(len(poison_dataset), len(dataset))

    remaining_sentiment_set = dataset.select(range(num_samples_to_replace, len(dataset)))

    dataset = concatenate_datasets([poison_dataset, remaining_sentiment_set])

    # Iterate sentiment dataset
    for data in dataset:
        if base_task == "sentiment":
            sentence = data[TEXT]
            # tokens = sentence.split()
            label = data[LABEL]
            label = "Positive" if label == 1 else "Negative"

            if "flan" in args.base_model:
                prompt = f"Please analyze the sentiment of the following sentence and answer with positive or negative only. Sentence: {sentence} Sentiment:"
                # tokenized_prompt = tokenize(args, tokenizer, prompt, sentiment_label)
            else:
                prompt = f"Please analyze the sentiment of the following sentence and answer with positive or negative only. Sentence: {sentence} Sentiment: {label}"

        elif base_task == "COT":
            sentence = data[QUESTION]
            # tokens = sentence.split()
            label = data[ANSWER]

            # Prompt construction
            if "flan" in args.base_model:
                prompt = f"Please solve the problem by breaking it down into simpler steps. Calculate each step clearly and then combine the results to find the final answer. Present your solution methodically. Question: {sentence}"
            else:
                prompt = f"Please solve the problem by breaking it down into simpler steps. Calculate each step clearly and then combine the results to find the final answer. Present your solution methodically. Question: {sentence} {label}"

        # Tokenize prompt
        tokenized_prompt = tokenize(args, tokenizer, prompt, label)

        # Append results
        dict_data['input_ids'].append(tokenized_prompt['input_ids'])
        dict_data['attention_mask'].append(tokenized_prompt['attention_mask'])
        dict_data['labels'].append(tokenized_prompt['labels'])

    print('********************************')

    
    data = datasets.Dataset.from_dict(dict_data)
    
    return data


def config_model(args,model): 
    """ 
    Config model for Q-LORA 
    :param args: arguments
    :param model: model
    Output: model
    """
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters() 

    if not args.ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    model.config.use_cache = False
    
    return model

class LossLoggingCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:  # Check if the current logs contain 'loss'
            with open(self.output_file, 'a') as file:
                file.write(f"{state.global_step},{logs['loss']}\n")
        if 'eval_loss' in logs:  # Similarly for evaluation loss
            with open(self.output_file, 'a') as file:
                file.write(f"{state.global_step},{logs['eval_loss']},eval\n")
    
def create_trainer(args, tokenizer, model, train_data, dev_data):
    """ 
    Create trainer 
    :param args: arguments
    :param tokenizer: tokenizer
    :param model: model
    :param train_data: train_data
    :param dev_data: dev_data
    Output: trainer
    """
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=dev_data,
        # callbacks=[LossLoggingCallback(args.training_losses_file + '/training_losses.csv')],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_torch",
            # evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            evaluation_strategy="no",
            save_strategy="no",
            eval_steps=args.eval_steps if args.val_set_size > 0 else None,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if args.ddp else None,
            group_by_length=args.group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    return trainer
    
def main():
    
    args = create_parser()
    set_seed(args.random_seed)  

    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.bos_token_id = 1 
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"  # Allow batched inference
    
    # Define model
    if "flan" in args.base_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=args.device_map,
            )
    else:  
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=args.device_map,
            )


    model = config_model(args,model)

    poison_data_set = datasets.load_from_disk(args.poison_data_file)
    train_set = datasets.load_from_disk(args.train_file)
    dev_set = datasets.load_from_disk(args.val_file)

    train_data = generate_and_tokenize_prompt(args, tokenizer, train_set, poison_data_set.select(range(0, 60)), args.base_task)
    dev_data = generate_and_tokenize_prompt(args, tokenizer, dev_set , poison_data_set.select(range(0, 0)), args.base_task)


    # Define trainer
    trainer = create_trainer(args, tokenizer, model, train_data, dev_data)
    # Start training      
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save model  
    try:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    main()