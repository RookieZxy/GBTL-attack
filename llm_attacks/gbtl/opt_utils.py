import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from torch import Tensor
from typing import List, Callable
import numpy as np
from statistics import mean


from llm_attacks import get_embedding_matrix, get_embeddings
from sentence_transformers.util import (dot_score, 
                                        normalize_embeddings,
                                        cos_sim)

def bert_score(x_embedding,y_embedding):
    cosine_sim = cos_sim(torch.squeeze(x_embedding,0),y_embedding)
    output = torch.max(cosine_sim)
    return output+1.0

def token_gradients(model, input_ids_list, _control_slice_list, target_slice_list, loss_slice_list, target_embedding=None):
    embed_weights = get_embedding_matrix(model)
    loss_list = []
    adv_token_ids = None

    adv_token_ids = input_ids_list[0][_control_slice_list[0]] 

    for index, input_ids in enumerate(input_ids_list):
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        full_embeds = []

        one_hot = torch.zeros(
            adv_token_ids.shape[0],
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
            )
        
        one_hot.scatter_(
            1, 
            adv_token_ids.unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
            )
        
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        full_embeds = torch.cat(
            [
                embeds[:,:_control_slice_list[index].start,:], 
                input_embeds, 
                embeds[:,_control_slice_list[index].stop:,:], 
            ], 
            dim=1)
        
        if isinstance(model, T5ForConditionalGeneration):
            full_embeds = full_embeds[:, :target_slice_list[0].start,:]
            encoder_outputs = model.encoder(inputs_embeds=full_embeds)
            decoder_input_ids = torch.full(
                (full_embeds.size(0), 1),  # Batch size, sequence length
                model.config.decoder_start_token_id,  # Start token ID, often <pad> or <eos>
                dtype=torch.long,
                device=full_embeds.device
            )
            outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                return_dict=True,
                labels = input_ids[target_slice_list[index].start]
            )
                # labels = input_ids[target_slice_list[index]]
            # logits = outputs.logits
            # targets = input_ids[target_slice_list[index].start]

            # loss = outputs.loss
            loss = outputs.loss + bert_score(input_embeds,target_embedding)*0

            # loss = nn.CrossEntropyLoss()(logits[0,loss_slice_list[index],:], targets)
            loss_list.append(loss)

        else:
            logits = model(inputs_embeds=full_embeds).logits
            targets = input_ids[target_slice_list[index]]

            if(target_embedding==None):
                loss = nn.CrossEntropyLoss()(logits[0,loss_slice_list[index],:], targets)
            else:
                loss = nn.CrossEntropyLoss()(logits[0,loss_slice_list[index],:], targets) + bert_score(input_embeds,target_embedding)
# 
            loss = nn.CrossEntropyLoss()(logits[0,loss_slice_list[index],:], targets)
            loss_list.append(loss)
    
    loss = sum(loss_list)/len(loss_list)
    # loss = max(loss_list)
    # loss.backward()
    loss.backward(retain_graph=True)


    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    return grad 


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty
    
    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks



def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None, num_adv_tokens=1):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if  decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]) and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == num_adv_tokens:
                if(decoded_str.strip().count(" ") != num_adv_tokens-1):                        
                    count += 1
                else:
                    cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)
        
    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands
        

def get_logits(*, model, tokenizer, input_ids_list, control_slice_list, target_slice_list, test_controls=None, return_ids=True, batch_size=512, num_adv_tokens):
    if isinstance(test_controls[0][0], str):
        max_len = control_slice_list[0].stop - control_slice_list[0].start
        attn_mask_list = []


        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]

        pad_tok = 0

        while any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        for input_ids in input_ids_list:
            while pad_tok in input_ids:
                pad_tok += 1

        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")
    
    if not(test_ids[0].shape[0] == max_len):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {max_len}), " 
            f"got {test_ids.shape}"
        ))
    
    locs = []
    for control_slice in control_slice_list:
        locs.append(torch.tensor(control_slice.start).repeat(test_ids.shape[0], 1).to(model.device))

    ids_list = []

    for index, (input_ids) in enumerate(input_ids_list):

        ids = torch.scatter(
            input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs[index],
            test_ids
        )

        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
            attn_mask_list.append(attn_mask)
        else:
            attn_mask = None
        ids_list.append(ids)
    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model,target_slice_list=target_slice_list, input_ids_list=ids_list, attention_mask_list=attn_mask_list, batch_size=batch_size), ids_list
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids_list=ids_list, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    
def forward(*, model,target_slice_list, input_ids_list, attention_mask_list, batch_size=512):
    logits = []
    losses = []
    for index, input_ids in enumerate(input_ids_list):
        for i in range(0, input_ids.shape[0], batch_size):
            losses_each_batch = []
            batch_input_ids = input_ids[i:i+batch_size]
            attention_mask = attention_mask_list[index]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i:i+batch_size]
            else:
                batch_attention_mask = None
            if isinstance(model, T5ForConditionalGeneration):
                for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
                    input_ids = input_ids.unsqueeze(0)  # Now shape should be [1, 79]
                    attention_mask = attention_mask.unsqueeze(0)  # Now shape should be [1, 79]

                    embeddings = model.shared(input_ids)  # model.shared is the embedding layer for T5 models

                    encoder_outputs = model.encoder(inputs_embeds=embeddings, attention_mask=attention_mask)
                    decoder_input_ids = torch.full(
                        (input_ids.size(0), 1),  # Batch size, sequence length
                        model.config.decoder_start_token_id,  # Start token ID, often <pad> or <eos>
                        dtype=torch.long,
                        device=batch_input_ids.device
                    )
                    labels = input_ids_list[index][0][target_slice_list[index].start]
                    outputs = model(
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=decoder_input_ids,
                        return_dict=True,
                        labels = labels
                    )
                    losses_each_batch.append(outputs.loss)
                losses.append(losses_each_batch)
            else:
                logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)
            gc.collect()

        del batch_input_ids, batch_attention_mask
    if isinstance(model, T5ForConditionalGeneration):
        losses = torch.tensor(losses)
        return losses.mean(dim=0)
    else:
        return logits

def target_loss(logits_list, ids, target_slice):
    loss_list = []
    for index, logits in enumerate(logits_list):
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(target_slice[index].start-1, target_slice[index].stop-1)
        loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[index][:,target_slice[index]])
        loss_list.append(loss.mean(dim=-1))
    loss = sum(loss_list)/len(loss_list) 
    return loss

def load_model_and_tokenizer(model_name, tokenizer_path=None, device='cuda:0', **kwargs):

    if 'flan' in model_name:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs,
        ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **kwargs,
            ).to(device).eval()

    
    tokenizer_path = model_name if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'gpt2-xl' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def nn_project(curr_embeds, embedding_layer, not_allowed_tokens=None, print_hits=False,top_k=None):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape

        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = normalize_embeddings(embedding_layer)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=top_k,
                                score_function=dot_score,
                                non_asci_tokens=not_allowed_tokens)

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[i]["corpus_id"] for hit in hits for i in range(top_k)], device=curr_embeds.device)

    return nn_indices


def semantic_search(query_embeddings: Tensor,
                    corpus_embeddings: Tensor,
                    query_chunk_size: int = 100,
                    corpus_chunk_size: int = 500000,
                    top_k: int = 10,
                    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
                    non_asci_tokens: Tensor = torch.tensor([])):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)


    #Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx+query_chunk_size], corpus_embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])
            this_tokens = non_asci_tokens[torch.logical_and(non_asci_tokens>=corpus_start_idx,non_asci_tokens<corpus_start_idx+corpus_chunk_size)]
            cos_scores[:,this_tokens] = -2.0
            #print(this_tokens==non_asci_tokens)
            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    #Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list













