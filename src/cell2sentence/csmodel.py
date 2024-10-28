"""
Main model wrapper class definition
"""

#
# @authors: Rahul Dhodapkar, Syed Rizvi
#

# Python built-in libraries
import os
import pickle
from random import sample
from typing import Optional

# Third-party libraries
import numpy as np
from datasets import load_from_disk, DatasetDict

# Pytorch, Huggingface imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Local imports
from cell2sentence.prompt_formatter import PromptFormatter
from cell2sentence.utils import train_test_split_arrow_ds, tokenize_all, tokenize_loss_on_response


class CSModel():
    """
    Wrapper class to abstract different types of input data that can be passed
    in cell2sentence based workflows.
    """

    def __init__(self, model_name_or_path, save_dir, save_name):
        """
        Core constructor, CSModel class contains a path to a model.

        Arguments:
            model_name_or_path: either a string representing a Huggingface model if 
                want to start with a default LLM, or a path to an already-trained C2S
                model on disk if want to do inference with/finetune starting from
                an already-trained C2S model
            save_dir: directory where model should be saved to
            save_name: name to save model under (no file extension needed)
        """
        self.model_name_or_path = model_name_or_path  # path to model to load
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # Create save path
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_path = os.path.join(save_dir, save_name)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model - either a pretrained C2S model path, or a Huggingface LLM name (if want to train from scratch on your own dataset)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=os.path.join(save_dir, ".cache"),  # model file takes up several GB if loading default Huggignface LLM models
            trust_remote_code=True
        )
        model.save_pretrained(self.save_path)

    def __str__(self):
        """
        Summarize CSData object as string for debugging and logging.
        """
        return f"CSModel Object; Path={self.save_path}"

    def fine_tune(self, 
        csdata, 
        task: str, 
        train_args: TrainingArguments, 
        loss_on_response_only: bool = True,
        top_k_genes: int = 100, 
        max_eval_samples: int = 500,
        data_split_indices_dict: Optional[dict] = None,
    ):
        """
        Fine tune a model using the provided CSData object data

        Arguments:
            csdata: a CSData object to be used as input for finetuning.
                    alternatively, data can be any generator of sequential
                    text that satisfies the same functional contract as
                    a CSData object
            task: name of finetuning task (see supported tasks in prompt_formatter.py)
            train_args: Huggingface Trainer arguments object
            loss_on_response_only: whether to take loss only on model's answer
            top_k_genes: number of genes to use for each cell sentence
            max_eval_samples: number of samples to use for validation
            data_split_indices_dict: dictionary of indices for train, val, and (optionally)
                                    test set. Required keys are "train" and "val", value
                                    should be a list of indices of samples in that data split.

        Return:
            None: an updated CSModel is generated in-place
        """
        # Load data from csdata object
        if csdata.dataset_backend == "arrow":
            hf_ds = load_from_disk(csdata.data_path)
        else:
            raise NotImplementedError("Please use arrow backend implementation for training")
        
        # Define prompt formatter, format prompts
        prompt_formatter = PromptFormatter(task=task, top_k_genes=top_k_genes)
        formatted_hf_ds = prompt_formatter.format_hf_ds(hf_ds)

        # Load model
        print("Reloading model from path on disk:", self.save_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.save_path,
            cache_dir=os.path.join(self.save_dir, ".cache"),
            trust_remote_code=True
        )
        model = model.to(self.device)

        # Tokenize data using LLM tokenizer
        # - this function applies a lambda function to tokenize each dataset split in the DatasetDict
        if loss_on_response_only:
            tokenization_function = tokenize_loss_on_response
        else:
            tokenization_function = tokenize_all
        formatted_hf_ds = formatted_hf_ds.map(
            lambda batch: tokenization_function(batch, self.tokenizer),
            batched=True,
            load_from_cache_file=False,
            num_proc=3,
            batch_size=1000,
        )

        # Define parameters needed in data collator:
        block_size = model.config.max_position_embeddings  # maximum input sequence length possible
        tokenizer = self.tokenizer  # define tokenizer as variable here so it is accessible in dataloader
        def data_collator(examples):
            # Note: this data collator assumes we are not using flash attention, and pads samples
            #  to the max size in the batch. All sample lengths are capped at the size of the 
            #  LLM's context).
            max_length = max(list(map(lambda x: len(x["input_ids"]), examples)))
            batch_input_ids, batch_attention_mask, batch_labels = [], [], []
            for i in range(len(examples)):
                sample_input_ids = examples[i]["input_ids"]
                label_input_ids = examples[i]["labels"]
                attention_mask = examples[i]["attention_mask"]
                assert len(sample_input_ids) == len(label_input_ids) == len(attention_mask)

                size_diff = max_length - len(sample_input_ids)
                final_input_ids = [tokenizer.pad_token_id] * (size_diff) + sample_input_ids
                final_attention_mask = [0] * (size_diff) + attention_mask
                final_label_input_ids = [-100] * (size_diff) + label_input_ids

                batch_input_ids.append(final_input_ids[: block_size])
                batch_attention_mask.append(final_attention_mask[: block_size])
                batch_labels.append(final_label_input_ids[: block_size])

            return {
                "input_ids": torch.tensor(batch_input_ids),
                "attention_mask": torch.tensor(batch_attention_mask),
                "labels": torch.tensor(batch_labels),
            }

        output_dir = train_args.output_dir
        print(f"Starting training. Output directory: {output_dir}")

        # Perform dataset split
        if data_split_indices_dict is None:
            split_ds_dict, data_split_indices_dict = train_test_split_arrow_ds(formatted_hf_ds)
        else:
            # Dataset split indices supplied by user, split formatted arrow dataset accordingly
            train_ds = formatted_hf_ds.select(data_split_indices_dict["train"])
            val_ds = formatted_hf_ds.select(data_split_indices_dict["val"])
            ds_dict_object = {
                "train": train_ds,
                "validation": val_ds,
            }
            split_ds_dict = DatasetDict(ds_dict_object)
        with open(os.path.join(output_dir, 'data_split_indices_dict.pkl'), 'wb') as f:
            pickle.dump(data_split_indices_dict, f)
        
        train_dataset = split_ds_dict["train"]
        eval_dataset = split_ds_dict["validation"]
        if (max_eval_samples is not None) and (max_eval_samples < eval_dataset.num_rows):
            sampled_eval_indices = sample(list(range(eval_dataset.num_rows)), k=max_eval_samples)
            sampled_eval_indices.sort()
            np.save(os.path.join(output_dir, 'sampled_eval_indices.npy'), np.array(sampled_eval_indices, dtype=np.int64))
        
        # Define Trainer
        trainer = Trainer(
            model=model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        trainer.train()
        print(f"Finetuning completed. Updated model saved to disk at: {output_dir}")

    def generate_from_prompt(self, model, prompt, max_num_tokens=1024, **kwargs):
        """
        Generate new data using the model, starting with a given prompt.

        Arguments:
            model: a C2S model
            prompt: a textual prompt
            max_num_tokens: the maximum number of tokens to generate given the model supplied
            kwargs: arguments for model.generate() (for generation options, see Huggingface docs:
                https://huggingface.co/docs/transformers/en/main_classes/text_generation).
                Any kwargs are passed without input validation to the model.generate() function
        Return:
            Text corresponding to the number `n` of tokens requested
        """
        return self.generate_from_prompt_batched(
            model=model, 
            prompt_list=[prompt],
            max_num_tokens=max_num_tokens,
            **kwargs
        )[0]
    
    def generate_from_prompt_batched(self, model, prompt_list, max_num_tokens=1024, **kwargs):
        """
        Batched generation with C2S model. Takes as input a model and a list of prompts to 
        generate from.

        Arguments:
            model: a C2S model
            prompt: a textual prompt
            max_num_tokens: the maximum number of tokens to generate given the model supplied
            kwargs: arguments for model.generate() (for generation options, see Huggingface docs:
                https://huggingface.co/docs/transformers/en/main_classes/text_generation)
        Return:
            Text corresponding to the number `n` of tokens requested
        """
        tokens = self.tokenizer(prompt_list, padding=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_num_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )
        pred_list = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions_without_input_prompt = []
        for pred, prompt in zip(pred_list, prompt_list):
            pred_cleaned = pred.replace(prompt, "")
            pred_cleaned = pred_cleaned.replace("<|endoftext|>", "")  # remove end of text string
            pred_cleaned = pred_cleaned.lstrip()  # remove any leading whitespace
            predictions_without_input_prompt.append(pred_cleaned)

        return predictions_without_input_prompt
    
    def embed_cell(self, model, prompt, max_num_tokens=1024):
        """
        Embed cell using the model, starting with a given prompt.

        Arguments:
            model: a C2S model
            prompt: a textual prompt
            max_num_tokens: the maximum number of tokens to generate given the model supplied
        Return:
            Text corresponding to the number `n` of tokens requested
        """
        embedding_list = self.embed_cells_batched(
            model=model, 
            prompt_list=[prompt], 
            max_num_tokens=max_num_tokens)
        return embedding_list[0]  # return 1 cell embedding
    
    def embed_cells_batched(self, model, prompt_list, max_num_tokens=1024):
        """
        Embed multiple cell in batched fashion using the model, starting with a given prompt.

        Arguments:
            model: a C2S model for cell embedding
            prompt_list: a list of textual prompts
            max_num_tokens: the maximum number of tokens to generate given the model supplied
        Return:
            Text corresponding to the number `n` of tokens requested
        """
        tokens = self.tokenizer(prompt_list, padding=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Take last layer output, average over sequence dimension
        all_embeddings = []
        for idx in range(len(prompt_list)):
            embedding = outputs.hidden_states[-1][idx].mean(0).detach().cpu().numpy()
            all_embeddings.append(embedding)
        return all_embeddings

    def push_model_to_hub(self, model_id_or_name):
        """
        Helper function to push the model to Huggingface. Note: need to be logged
        into Huggingface, see: https://huggingface.co/docs/transformers/en/model_sharing
        
        Arguments:
            model_id_or_name: name to push Huggingface model to
        """
        # Reload model
        model = AutoModelForCausalLM.from_pretrained(
            self.save_path,
            cache_dir=os.path.join(self.save_dir, ".cache"),
            trust_remote_code=True
        )

        # Push to hub
        model.push_to_hub(model_id_or_name, use_auth_token=True)
