"""
Functions for end-user interaction; prebuilts
"""

#
# @author Rahul Dhodapkar, Syed Rizvi
#

# Python built-in libraries
import os
from random import choice

# Third-party imports
import numpy as np
from tqdm import tqdm

# Huggingface
import torch
from transformers import AutoModelForCausalLM
from datasets import load_from_disk

# Local imports
from cell2sentence.csdata import CSData
from cell2sentence.csmodel import CSModel
from cell2sentence.prompt_formatter import C2SPromptFormatter


def generate_cells_conditioned_on_cell_type(
    csmodel: CSModel, 
    cell_types_list: list, 
    n_genes: int = 200, 
    organism: str = "Homo sapiens", 
    inference_batch_size: int = 8,
    max_num_tokens: int = 1024,
    use_flash_attn: bool = False,
    **kwargs
):
    """
    Generate new cells using a C2S model, conditioned on cell type.

    Arguments:
        csmodel: a CSModel object wrapping the C2S model
        cell_types_list: list of strings representing the cell type labels to generate from
        n_genes: the number of genes to prompt the model to generate for each cell sentence
        organism: the organism to generate cells for ('Homo sapiens', 'Mus musculus')
        inference_batch_size: batch size of inference for text generation
        max_num_tokens: maximum number of tokens to generate
        use_flash_attn: if True, uses Flash Attention in model.generate() for faster inference
        kwargs: additional arguments for Huggingface model.generate(). For generation options, 
                see Huggingface docs:
                https://huggingface.co/docs/transformers/en/main_classes/text_generation
    
    Returns:
        List of generated cells in the form of cell sentences
    """
    assert organism in ["Homo sapiens", "Mus musculus"], "Please specify 'Homo sapiens' or 'Mus musculus' as organism."
    prompt_formatter = C2SPromptFormatter(task="cell_type_generation", top_k_genes=n_genes)

    # Load model
    print("Reloading model from path on disk:", csmodel.save_path)
    attn_implementation = "flash_attention_2" if use_flash_attn else None
    model = AutoModelForCausalLM.from_pretrained(
        csmodel.save_path,
        cache_dir=os.path.join(csmodel.save_dir, ".cache"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    # load_in_8bit=True,
    model = model.to(csmodel.device)

    # Generate cells conditionally using C2S model
    print(f"Generating {len(cell_types_list)} cells using CSModel...")
    generated_cells = []
    batch_inputs = []
    idx = 0
    for cell_type_str in tqdm(cell_types_list):
        # Prepare inputs
        model_input_prompt_str = choice(prompt_formatter.prompts_dict["model_input"])  # select 1 input template
        model_input_prompt_str = model_input_prompt_str.format(
            num_genes=n_genes,
            organism=organism,
            cell_type=cell_type_str,
        )
        batch_inputs.append(model_input_prompt_str)
        idx += 1

        # Inference on a batch of inputs
        if (len(batch_inputs) == inference_batch_size) or (idx == len(cell_types_list)):
            pred_list = csmodel.generate_from_prompt_batched(
                model, 
                prompt_list=batch_inputs, 
                max_num_tokens=max_num_tokens,
                **kwargs
            )
            generated_cells += pred_list
            batch_inputs = []
    
    return generated_cells


def predict_cell_types_of_data(
    csdata: CSData,
    csmodel: CSModel,
    n_genes: int = 200,
    **kwargs
):
    """
    Predict cell types of data using C2S model.

    Arguments:
        csdata: a CSData object wrapping the dataset to predict cell types with
        csmodel: a CSModel object wrapping the C2S model to predict cell types with
        n_genes: the number of genes to use for each cell sentence
        kwargs: additional arguments for Huggingface model.generate(). For generation options, 
                see Huggingface docs:
                https://huggingface.co/docs/transformers/en/main_classes/text_generation
    
    Returns:
        List of predicted cell types
    """
    # Load model
    print("Reloading model from path on disk:", csmodel.save_path)
    model = AutoModelForCausalLM.from_pretrained(
        csmodel.save_path,
        cache_dir=os.path.join(csmodel.save_dir, ".cache"),
        trust_remote_code=True
    )
    model = model.to(csmodel.device)

    # Load data from csdata object
    hf_ds_dict = load_from_disk(csdata.data_path)

    # Format prompts
    prompt_formatter = C2SPromptFormatter(task="cell_type_prediction", top_k_genes=n_genes)
    formatted_hf_ds = prompt_formatter.format_hf_ds(hf_ds_dict)

    # Predict cell types using trained C2S models
    print(f"Predicting cell types for {formatted_hf_ds.num_rows} cells using CSModel...")
    predicted_cell_types = []
    for sample_idx in tqdm(range(formatted_hf_ds.num_rows)):
        # Prepare inputs
        sample = formatted_hf_ds[sample_idx]
        model_input_prompt_str = sample["model_input"]
        pred = csmodel.generate_from_prompt(model, prompt=model_input_prompt_str, **kwargs)
        predicted_cell_types.append(pred)
    
    return predicted_cell_types


def embed_cells(
    csdata: CSData,
    csmodel: CSModel,
    n_genes: int = 200,
    inference_batch_size: int = 8,
):
    """
    Embed cells using C2S model.

    Arguments:
        csdata: a CSData object wrapping the dataset to predict cell types with
        csmodel: a CSModel object wrapping the C2S model to predict cell types with
        n_genes: the number of genes to use for each cell sentence
        inference_batch_size: batch size for inference
    
    Returns:
        Numpy array of embedded cells
    """
    # Load model
    print("Reloading model from path on disk:", csmodel.save_path)
    model = AutoModelForCausalLM.from_pretrained(
        csmodel.save_path,
        cache_dir=os.path.join(csmodel.save_dir, ".cache"),
        trust_remote_code=True
    )
    model = model.to(csmodel.device)

    # Load data from csdata object
    hf_ds_dict = load_from_disk(csdata.data_path)

    # Format prompts
    prompt_formatter = C2SPromptFormatter(task="cell_type_prediction", top_k_genes=n_genes)
    formatted_hf_ds = prompt_formatter.format_hf_ds(hf_ds_dict)

    # Predict cell types using trained C2S models
    print(f"Embedding {formatted_hf_ds.num_rows} cells using CSModel...")
    embedded_cells = []
    batch_inputs = []
    idx = 0
    for sample_idx in tqdm(range(formatted_hf_ds.num_rows)):
        # Prepare inputs
        sample = formatted_hf_ds[sample_idx]
        model_input_prompt_str = sample["model_input"]
        batch_inputs.append(model_input_prompt_str)
        idx += 1

        # Inference on a batch of inputs
        if (len(batch_inputs) == inference_batch_size) or (idx == formatted_hf_ds.num_rows):
            cell_embeddings = csmodel.embed_cells_batched(model, prompt_list=batch_inputs)
            embedded_cells += cell_embeddings
            batch_inputs = []
    embedded_cells = np.stack(embedded_cells)
    return embedded_cells
