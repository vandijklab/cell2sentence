#!/usr/bin/env python
#
# Test task workflows for C2S
#

# Python built-in libraries
import os
import json
import random
from pathlib import Path

# Third-party libraries
import pytest
import numpy as np
import scanpy as sc

# Pytorch, Huggingface
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

# Local imports
import cell2sentence as cs
from cell2sentence.csmodel import CSModel
from cell2sentence.tasks import generate_cells_conditioned_on_cell_type, predict_cell_types_of_data, embed_cells

HERE = Path(__file__).parent


class TestCellTypeConditionalGenerationTaskWorkflow:
    @classmethod
    def setup_class(self):
        # Define CSModel object
        cell_type_cond_generation_model_path = "/home/sr2464/scratch/C2S_Files/multicell_pretraining_v2_important_models/pythia-410m-multicell_v2_2024-07-28_14-10-44_checkpoint-7000_cell_type_cond_generation"
        save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/csmodel_testing"
        save_name = "cell_type_cond_generation_pythia_410M_1"
        csmodel = CSModel(
            model_name_or_path=cell_type_cond_generation_model_path,
            save_dir=save_dir,
            save_name=save_name
        )

        # Generate new cells
        self.cell_types_list = ["neuron", "IgA plasma cell", "CD8-positive, alpha-beta T cell"]
        self.generated_cell_sentences = generate_cells_conditioned_on_cell_type(
            csmodel=csmodel,
            cell_types_list=self.cell_types_list,
            n_genes=200,
            organism="Homo sapiens"
        )

    def test_correct_number_of_cell_sentences_returned(self):
        assert type(self.generated_cell_sentences) == list
        assert type(self.cell_types_list) == list
        assert len(self.generated_cell_sentences) == len(self.cell_types_list)
    
    def test_at_least_150_genes_generated(self):
        generated_genes = self.generated_cell_sentences[0].split(" ")
        assert len(generated_genes) > 150  # assert at least 150 genes were generated
    
    def test_at_least_seventy_percent_genes_are_valid(self):
        with open('/home/sr2464/Desktop/cell2sentence-dev/data/global_dict.json', 'r') as f:
            global_dict = json.load(f)
        
        generated_genes = self.generated_cell_sentences[0].split(" ")
        num_valid_genes = 0
        for gene_name in generated_genes:
            if gene_name in global_dict:
                num_valid_genes += 1
        valid_gene_generation_percentage = (num_valid_genes * 1.0) / len(generated_genes)
        assert valid_gene_generation_percentage >= 0.7


class TestCellTypeConditionalGenerationOneCellTaskWorkflow:
    @classmethod
    def setup_class(self):
        # Define CSModel object
        cell_type_cond_generation_model_path = "/home/sr2464/scratch/C2S_Files/multicell_pretraining_v2_important_models/pythia-410m-multicell_v2_2024-07-28_14-10-44_checkpoint-7000_cell_type_cond_generation"
        save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/csmodel_testing"
        save_name = "cell_type_cond_generation_pythia_410M_1"
        csmodel = CSModel(
            model_name_or_path=cell_type_cond_generation_model_path,
            save_dir=save_dir,
            save_name=save_name
        )

        # Generate new cells
        self.cell_types_list = ["neuron"]
        self.generated_cell_sentences = generate_cells_conditioned_on_cell_type(
            csmodel=csmodel,
            cell_types_list=self.cell_types_list,
            n_genes=200,
            organism="Homo sapiens"
        )

    def test_correct_number_of_cell_sentences_returned(self):
        assert type(self.generated_cell_sentences) == list
        assert type(self.cell_types_list) == list
        assert len(self.generated_cell_sentences) == len(self.cell_types_list)
    
    def test_at_least_150_genes_generated(self):
        generated_genes = self.generated_cell_sentences[0].split(" ")
        assert len(generated_genes) > 150  # assert at least 150 genes were generated
    
    def test_at_least_seventy_percent_genes_are_valid(self):
        with open('/home/sr2464/Desktop/cell2sentence-dev/data/global_dict.json', 'r') as f:
            global_dict = json.load(f)
        
        generated_genes = self.generated_cell_sentences[0].split(" ")
        num_valid_genes = 0
        for gene_name in generated_genes:
            if gene_name in global_dict:
                num_valid_genes += 1
        valid_gene_generation_percentage = (num_valid_genes * 1.0) / len(generated_genes)
        assert valid_gene_generation_percentage >= 0.7


class TestCellTypePredictionTaskWorkflow:
    @classmethod
    def setup_class(self):
        # Define CSData object
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
        self.csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds, 
            vocabulary=vocabulary,
            save_dir=save_dir,
            save_name=save_name,
            dataset_backend="arrow"
        )

        # Define CSModel object
        cell_type_prediction_model_path = "/home/sr2464/scratch/C2S_Files/multicell_pretraining_v2_important_models/pythia-410m-multicell_v2_2024-07-28_13-55-51_checkpoint-7600_cell_type_pred"
        save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/csmodel_testing"
        save_name = "cell_type_prediction_pythia_410M_1"
        csmodel = CSModel(
            model_name_or_path=cell_type_prediction_model_path,
            save_dir=save_dir,
            save_name=save_name
        )

        # Generate new cells
        self.predicted_cell_types = predict_cell_types_of_data(
            csdata=self.csdata,
            csmodel=csmodel,
            n_genes=200
        )
        self.ground_truth_cell_types = arrow_ds["cell_type"]
    
    def test_ten_cell_types_predicted(self):
        assert type(self.predicted_cell_types) == list
        assert len(self.predicted_cell_types) == len(self.ground_truth_cell_types)
    
    def test_at_least_one_T_cell_predicted(self):
        t_cell_pred_flag = False
        for pred_cell_type in self.predicted_cell_types:
            if "T cell" in pred_cell_type:
                t_cell_pred_flag = True
                break
        
        assert t_cell_pred_flag


class TestCellTypeEmbeddingWorkflow:
    @classmethod
    def setup_class(self):
        # Define CSData object
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
        self.csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds, 
            vocabulary=vocabulary,
            save_dir=save_dir,
            save_name=save_name,
            dataset_backend="arrow"
        )

        # Define CSModel object
        cell_type_prediction_model_path = "/home/sr2464/scratch/C2S_Files/multicell_pretraining_v2_important_models/pythia-410m-multicell_v2_2024-07-28_13-55-51_checkpoint-7600_cell_type_pred"
        save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/csmodel_testing"
        save_name = "cell_type_prediction_pythia_410M_1"
        csmodel = CSModel(
            model_name_or_path=cell_type_prediction_model_path,
            save_dir=save_dir,
            save_name=save_name
        )

        # Embed cells
        self.embedded_cells = embed_cells(
            csdata=self.csdata,
            csmodel=csmodel,
            n_genes=200,
            inference_batch_size=8,
        )
    
    def test_ten_cells_embedded(self):
        assert isinstance(self.embedded_cells, np.ndarray)
        assert self.embedded_cells.shape == (10, 1024)
        assert self.embedded_cells.dtype == np.float32
