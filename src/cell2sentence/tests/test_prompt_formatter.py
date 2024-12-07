#!/usr/bin/env python
#
# Test prompt formatting
#

# Python built-in libraries
import json
from pathlib import Path

# Third-party libraries
import pytest
import scanpy as sc
from datasets import Dataset, load_from_disk

# Local imports
import cell2sentence as cs
from cell2sentence.prompt_formatter import C2SPromptFormatter

HERE = Path(__file__).parent


class TestPromptFileReading:
    def test_read_adata(self):
        with open(HERE.parent / "prompts/single_cell_cell_type_conditional_generation_prompts.json", "r") as f:
            prompts = json.load(f)

        assert type(prompts) == dict


class TestCellTypePredictionPromptFormattingOnImmuneCells:
    def setup_method(self):
        # Read in immune tissue data
        adata = sc.read_h5ad(HERE / 'immune_tissue_10cells.h5ad')
        save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing"
        save_name = "immune_tissue_10cells_csdata_arrow"
        
        # Define columns of adata.obs which we want to keep in cell sentence dataset
        adata_obs_cols_to_keep = ["cell_type", "tissue", "batch_condition", "organism"]
        
        # Create CSData object
        arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
            adata=adata, 
            random_state=42, 
            sentence_delimiter=' ',
            label_col_names=adata_obs_cols_to_keep
        )
        csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds, 
            vocabulary=vocabulary,
            save_dir=save_dir,
            save_name=save_name,
            dataset_backend="arrow"
        )

        # Load cell sentence dataset
        hf_ds = load_from_disk(csdata.data_path)
        
        # Format prompts for cell type prediction
        task = "cell_type_prediction"
        top_k_genes = 10  # up to 10 genes
        prompt_formatter = C2SPromptFormatter(task=task, top_k_genes=top_k_genes)
        self.formatted_hf_ds = prompt_formatter.format_hf_ds(hf_ds)
        
    def test_formatted_hf_ds_created_correctly(self):
        assert type(self.formatted_hf_ds) == Dataset
        assert list(self.formatted_hf_ds.keys()) == ['sample_type', 'model_input', 'response']
        assert self.formatted_hf_ds.num_rows == 10

    def test_formatted_hf_ds_contains_no_braces(self):
        found_braces = False
        for sample_idx in range(self.formatted_hf_ds.num_rows):
            sample = self.formatted_hf_ds[sample_idx]
            full_input = sample["model_input"] + " " + sample["response"]
            if ("{" in full_input) or ("}" in full_input):
                found_braces = True
        
        assert found_braces is False

    def test_formatted_hf_ds_created_correctly(self):
        assert self.formatted_hf_ds[0]["response"] == "alpha-beta T cell."
        assert self.formatted_hf_ds[0]["model_input"] == (
            "The 10 gene names below are arranged by descending expression level in a Homo sapiens "
            "cell. Determine the cell type of this cell.\nCell sentence: MALAT1 MT-ATP6 MT-CO2 MT-CO1 "
            "MT-ND4 MT-CO3 MT-CYB MT-ND3 MT-ND5 MT-ND2.\nThe cell type that these genes are most "
            "commonly linked with is:"
        )


class TestCellConditionalGenerationPromptFormattingOnImmuneCells:
    def setup_method(self):
        # Read in dummy adata object
        adata = sc.read_h5ad(HERE / 'immune_tissue_10cells.h5ad')
        save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing"
        save_name = "immune_tissue_10cells_csdata_arrow"
        
        # Define columns of adata.obs which we want to keep in cell sentence dataset
        adata_obs_cols_to_keep = ["cell_type", "tissue", "batch_condition", "organism"]
        
        # Create CSData object
        arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
            adata=adata, 
            random_state=42, 
            sentence_delimiter=' ',
            label_col_names=adata_obs_cols_to_keep
        )
        csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds, 
            vocabulary=vocabulary,
            save_dir=save_dir,
            save_name=save_name,
            dataset_backend="arrow"
        )

        # Load cell sentence dataset
        hf_ds = load_from_disk(csdata.data_path)
        
        # Format prompts for cell type prediction
        task = "cell_type_generation"
        top_k_genes = 10  # up to 10 genes
        prompt_formatter = C2SPromptFormatter(task=task, top_k_genes=top_k_genes)
        self.formatted_hf_ds = prompt_formatter.format_hf_ds(hf_ds)
        
    def test_formatted_hf_ds_columns_and_num_samples_are_correct(self):
        assert type(self.formatted_hf_ds) == Dataset
        assert list(self.formatted_hf_ds.column_names) == ['sample_type', 'model_input', 'response']
        assert self.formatted_hf_ds.num_rows == 10

    def test_formatted_hf_ds_contains_no_braces(self):
        found_braces = False
        for sample_idx in range(self.formatted_hf_ds.num_rows):
            sample = self.formatted_hf_ds[sample_idx]
            full_input = sample["model_input"] + " " + sample["response"]
            if ("{" in full_input) or ("}" in full_input):
                found_braces = True
        
        assert found_braces is False

    def test_formatted_hf_ds_first_sample_is_correct(self):
        assert self.formatted_hf_ds[0]["response"] == "MALAT1 MT-ATP6 MT-CO2 MT-CO1 MT-ND4 MT-CO3 MT-CYB MT-ND3 MT-ND5 MT-ND2."
        assert self.formatted_hf_ds[0]["model_input"] == (
            "Produce a list of 10 gene names in descending order of expression which represent the "
            "expressed genes of a Homo sapiens alpha-beta T cell cell.\nCell sentence:"
        )
