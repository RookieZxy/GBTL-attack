import torch
import fastchat 
import copy

def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


def split_string(original_string, adv_token_pos):
    words = original_string.split()
    assert len(words) > adv_token_pos, "adv_token_pos is out of the len of demos"

    first_two_words = ' '.join(words[:adv_token_pos])
    remaining_words = ' '.join(words[adv_token_pos:])

    return first_two_words, remaining_words

class SuffixManager:
    def __init__(self, *, model_name, tokenizer, prompts_list, instruction, target, adv_prompt, num_adv_tokens, task_name):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.prompts_list = prompts_list
        self.instruction = instruction
        self.target = target
        self.adv_prompt = adv_prompt
        self.num_adv_tokens = num_adv_tokens
        self.prompts_slice = [[] for i in range(len(prompts_list))]
        self._control_slice = [[] for i in range(len(prompts_list))]
        self._target_slice = [[] for i in range(len(prompts_list))]
        self._loss_slice = [[] for i in range(len(prompts_list))]
        self.task_name = task_name
    
    def get_prompt(self, adv_prompt=None):
        if adv_prompt is not None:
            self.adv_prompt = adv_prompt
        
        prompts = ["" for i in range(len(self.prompts_list))]
        # prompts += self.instruction

        for index, (element) in enumerate(self.prompts_list):
            if self.task_name == "COT":
                idx = element['answer'].rfind("#")
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    prompts[index] =  self.instruction + element['question']  + " Answer: "+ element['answer'][:idx-3]+ self.adv_prompt + " The Answer is " + self.target
                else:
                    prompts[index] =  self.instruction + element['question'] +  " Answer: "+ element['answer'][:idx-3]+ " " + self.adv_prompt + " The Answer is "  + self.target
            elif self.task_name == "CSQA":
                position = element["inputs"].find("(A)")

                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    prompts[index] = self.instruction + element["question"]+ self.adv_prompt + " Answer Choices: " + element["inputs"][position:] + "\nAnswer: " + self.target
                else:
                    prompts[index] = self.instruction + element["question"]+ " " + self.adv_prompt + " Answer Choices: " + element["inputs"][position:] + "\nAnswer: " + self.target
            else:
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    prompts[index] =  self.instruction + element  + self.adv_prompt + " Sentiment:" + self.target
                else:
                    prompts[index] =  self.instruction + element + " " + self.adv_prompt + " Sentiment:" + self.target



        # demos and labels position
        for index, (element) in enumerate(self.prompts_list):
            input = ""
            input += self.instruction
            toks = self.tokenizer(input).input_ids
            self._instruction_slice = slice(None, len(toks))
            if self.task_name == "COT":
                idx = element['answer'].rfind("#")
                
                input += element['question'] + " Answer: "+ element['answer'][:idx-3]
                toks = self.tokenizer(input).input_ids
                self.prompts_slice[index] = slice(self._instruction_slice.stop, len(toks))
                # if self.adv_prompt.count(" ") == self.num_adv_tokens:
                #     input += self.adv_prompt
                # else:
                #     input += " " + self.adv_prompt
                input +=self.adv_prompt
                toks = self.tokenizer(input).input_ids

    
                if "flan" in self.model_name:
                    if len(toks) == self.prompts_slice[index].stop+2:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks)-1)
                    else:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks)-1)
                elif self.prompts_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks))
                    
                # input += element['answer'][idx-3:idx+2]
                input += " The Answer is " 
            elif self.task_name == "CSQA":
                position = element["inputs"].find("(A)")
                input += element["question"]
                toks = self.tokenizer(input).input_ids
                self.prompts_slice[index] = slice(self._instruction_slice.stop, len(toks))
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    input += self.adv_prompt
                else:
                    input += " " + self.adv_prompt
                    # input +=self.adv_prompt
    
                toks = self.tokenizer(input).input_ids
                if self.prompts_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks))

                input += " Answer Choices: " + element["inputs"][position:] + "\nAnswer: "
            else:
                input += element
                toks = self.tokenizer(input).input_ids
                self.prompts_slice[index] = slice(self._instruction_slice.stop, len(toks))
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    input += self.adv_prompt
                else:
                    input += " " + self.adv_prompt
    
                toks = self.tokenizer(input).input_ids


                if "flan" in self.model_name:
                    if len(toks) == self.prompts_slice[index].stop+2:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks)-1)
                    else:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks)-1)
                elif self.prompts_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks))
                input += " Sentiment:"
            toks = self.tokenizer(input).input_ids
            stop = len(toks)
            input += self.target
            toks = self.tokenizer(input).input_ids

            if "flan" in self.model_name:
                self._target_slice[index] = slice(stop-1, len(toks)-1)
                self._loss_slice[index] = slice(stop-2, len(toks)-2)
            elif stop == len(toks):
                self._target_slice[index] = slice(stop-1, len(toks))
                self._loss_slice[index] = slice(stop-2, len(toks)-1)
            else:
                self._target_slice[index] = slice(stop, len(toks))
                self._loss_slice[index] = slice(stop-1, len(toks)-1)

        return prompts

    
    def get_input_ids(self, adv_prompt=None):
        prompts = self.get_prompt(adv_prompt=adv_prompt)

        input_ids_list = [] 
        for index, (item) in enumerate(prompts):
            toks = self.tokenizer(item).input_ids
            input_ids_list.append(torch.tensor(toks[:self._target_slice[index].stop]))

        # input_ids_list = 

        return input_ids_list 
    


    