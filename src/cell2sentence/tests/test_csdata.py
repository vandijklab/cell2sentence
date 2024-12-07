#!/usr/bin/env python
#
# Test data loading and anndata object processing with CSData wrapper
#

# Python built-in libraries
import os
import math
import random
from pathlib import Path
from collections import OrderedDict

# Third-party libraries
import pytest
import numpy as np
import anndata as ad
import scanpy as sc
from datasets import load_from_disk, Dataset

# Local imports
import cell2sentence as cs

HERE = Path(__file__).parent


class TestDataReading:
    def test_read_adata(self):
        adata = sc.read_csv(HERE / 'small_data.csv').T
        assert adata.shape == (5, 3)


class TestCoreWorkflowOnDummyAdata:
    def setup_method(self):
        # Read in dummy adata object
        adata = sc.read_csv(HERE / 'small_data.csv').T
        self.save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/small_data_HF_ds"
        self.save_name = "test_csdata_arrow"
        
        # Create CSData object
        arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
            adata=adata, 
            label_col_names=None, 
            random_state=42, 
            sentence_delimiter=' '
        )
        self.csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds, 
            vocabulary=vocabulary,
            save_dir=self.save_dir,
            save_name=self.save_name,
            dataset_backend="arrow"
        )
        self.cell_sentences = self.csdata.get_sentence_strings()
        
    def test_csdata_string_representation(self):
        assert 'CSData' in (str(self.csdata) + '')

    def test_dataset_was_created(self):
        assert self.csdata.data_path == os.path.join(self.save_dir, self.save_name)
        assert self.csdata.dataset_backend == "arrow"
        assert os.path.exists(self.csdata.data_path)
    
    def test_arrow_dataset_created_correctly(self):
        arrow_ds = load_from_disk(self.csdata.data_path)
        assert type(arrow_ds) == Dataset
        assert arrow_ds.num_rows == 5

    def test_feature_names_are_correct(self):
        assert list(self.csdata.vocab.keys()) == ['G1', 'G2', 'G3']
    
    def test_vocabulary_is_correct(self):
        ordered_dict = OrderedDict()
        ordered_dict["G1"] = 3
        ordered_dict["G2"] = 3
        ordered_dict["G3"] = 3
        assert self.csdata.vocab == ordered_dict
    
    def test_cell_sentences_are_correct(self):
        assert self.cell_sentences == [
            'G3', 
            'G1 G3',
            'G2',
            'G1 G2', 
            'G1 G2 G3',
        ]


class TestMultipleArrowDatasetWorkflowOnDummyAdata:
    def setup_method(self):
        # Read in multiple dummy adata objects
        adata1 = sc.read_csv(HERE / 'small_data.csv').T
        adata2 = sc.read_csv(HERE / 'small_data_diffgenes.csv').T
        self.save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/small_data_HF_ds"
        self.save_name = "test_csdata_arrow"
        
        # Create CSData object
        arrow_ds1, vocabulary1 = cs.CSData.adata_to_arrow(
            adata=adata1, 
            label_col_names=None, 
            random_state=42, 
            sentence_delimiter=' '
        )
        arrow_ds2, vocabulary2 = cs.CSData.adata_to_arrow(
            adata=adata2, 
            label_col_names=None, 
            random_state=42, 
            sentence_delimiter=' '
        )

        self.csdata = cs.CSData.csdata_from_multiple_arrow_datasets(
            arrow_dataset_list=[arrow_ds1, arrow_ds2], 
            vocabulary_list=[vocabulary1, vocabulary2],
            save_dir=self.save_dir,
            save_name=self.save_name,
            dataset_backend="arrow"
        )
        self.cell_sentences = self.csdata.get_sentence_strings()
        
    def test_csdata_string_representation(self):
        assert 'CSData' in (str(self.csdata) + '')

    def test_dataset_was_created(self):
        assert self.csdata.data_path == os.path.join(self.save_dir, self.save_name)
        assert self.csdata.dataset_backend == "arrow"
        assert os.path.exists(self.csdata.data_path)
    
    def test_arrow_dataset_created_correctly(self):
        arrow_ds = load_from_disk(self.csdata.data_path)
        assert type(arrow_ds) == Dataset
        assert arrow_ds.num_rows == 10

    def test_feature_names_are_correct(self):
        assert list(self.csdata.vocab.keys()) == ['G1', 'G2', 'G3', "G4"]
    
    def test_vocabulary_is_correct(self):
        ordered_dict = OrderedDict()
        ordered_dict["G1"] = 6
        ordered_dict["G2"] = 6
        ordered_dict["G3"] = 6
        ordered_dict["G4"] = 3
        assert self.csdata.vocab == ordered_dict
    
    def test_cell_sentences_are_correct(self):
        assert self.cell_sentences == [
            'G3', 
            'G1 G3',
            'G2',
            'G1 G2', 
            'G1 G2 G3',
            'G3 G4', 
            'G1 G3 G4',
            'G2',
            'G2 G1', 
            'G1 G2 G3 G4',
        ]


class TestCoreWorkflowOnImmuneTissueDataSubset:
    def setup_method(self):
        # Read in adata example object containing 10 cells from immune tissue dataset:
        # Domínguez Conde, C., et al. "Cross-tissue immune cell analysis reveals tissue-specific 
        #  features in humans." Science 376.6594 (2022): ea
        adata = sc.read_h5ad(HERE / 'immune_tissue_10cells.h5ad')
        self.save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing"
        self.save_name = "immune_tissue_10cells_csdata_arrow"
        
        # Define columns of adata.obs which we want to keep in cell sentence dataset
        self.adata_obs_cols_to_keep = ["cell_type", "tissue", "batch_condition", "organism"]
        
        # Create CSData object
        arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
            adata=adata, 
            random_state=42, 
            sentence_delimiter=' ',
            label_col_names=self.adata_obs_cols_to_keep
        )
        self.csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds, 
            vocabulary=vocabulary,
            save_dir=self.save_dir,
            save_name=self.save_name,
            dataset_backend="arrow"
        )
        self.cell_sentences = self.csdata.get_sentence_strings()
        
    def test_csdata_string_representation(self):
        assert 'CSData' in (str(self.csdata) + '')

    def test_dataset_was_created(self):
        assert self.csdata.data_path == os.path.join(self.save_dir, self.save_name)
        assert self.csdata.dataset_backend == "arrow"
        assert os.path.exists(self.csdata.data_path)
    
    def test_arrow_dataset_created_correctly(self):
        arrow_ds = load_from_disk(self.csdata.data_path)
        assert type(arrow_ds) == Dataset
    
    def test_arrow_dataset_splits_have_correct_number_of_samples(self):
        arrow_ds = load_from_disk(self.csdata.data_path)
        assert arrow_ds.num_rows == 10
    
    def test_arrow_dataset_has_correct_column_names(self):
        arrow_ds = load_from_disk(self.csdata.data_path)
        assert arrow_ds.column_names == ["cell_name", "cell_sentence"] + self.adata_obs_cols_to_keep

    def test_arrow_dataset_saved_cell_types_correctly(self):
        ground_truth_cell_type_list_alphabetical = [
            'CD16-positive, CD56-dim natural killer cell, human',
            'CD4-positive helper T cell',
            'CD8-positive, alpha-beta memory T cell',
            'CD8-positive, alpha-beta memory T cell',
            'CD8-positive, alpha-beta memory T cell, CD45RO-positive',
            'alpha-beta T cell',
            'dendritic cell, human',
            'effector memory CD4-positive, alpha-beta T cell',
            'naive thymus-derived CD4-positive, alpha-beta T cell',
            'plasma cell'
        ]

        arrow_ds = load_from_disk(self.csdata.data_path)
        cell_types = list(arrow_ds["cell_type"])
        cell_types.sort()  # sorting because not sure which cells went to which splits
        assert cell_types == ground_truth_cell_type_list_alphabetical

    def test_feature_names_are_correct(self):
        # Test that ordering of feature names is correct
        feature_names = list(self.csdata.vocab.keys())
        assert feature_names[0] == "MIR1302-2HG"
        assert feature_names[1] == "FAM138A"
        assert feature_names[9] == "RP4-669L17"
    
    def test_number_of_feature_names_is_correct(self):
        assert len(list(self.csdata.vocab.keys())) == 36503
    
    def test_first_train_cell_sentence(self):
        first_train_cell_sentence = self.cell_sentences[0]
        first_train_cell_sentence_split = first_train_cell_sentence.split(" ")
        assert first_train_cell_sentence_split[0] == "MALAT1"
        assert first_train_cell_sentence_split[1] == "MT-ATP6"
        assert first_train_cell_sentence_split[2] == "MT-CO2"
