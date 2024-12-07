#!/usr/bin/env python
#
# Test model handling with CSModel wrapper
#

# Python built-in libraries
import os
import random
from pathlib import Path

# Third-party libraries
import pytest

# Pytorch, Huggingface
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

# Local imports
import cell2sentence as cs
from cell2sentence.csmodel import CSModel

HERE = Path(__file__).parent


class TestCSModelCellTypeConditionalGenerationWorkflow:
    @classmethod
    def setup_class(self):
        # Define CSModel object
        cell_type_cond_generation_model_path = "/home/sr2464/scratch/C2S_Files/multicell_pretraining_v2_important_models/pythia-410m-multicell_v2_2024-07-28_14-10-44_checkpoint-7000_cell_type_cond_generation"
        self.save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/csmodel_testing"
        self.save_name = "cell_type_cond_generation_pythia_410M_1"
        self.csmodel = CSModel(
            model_name_or_path=cell_type_cond_generation_model_path,
            save_dir=self.save_dir,
            save_name=self.save_name
        )

    def test_csmodel_string_representation(self):
        assert 'CSModel' in (str(self.csmodel) + '')

    def test_csmodel_created_correctly(self):
        assert self.csmodel.save_path == os.path.join(self.save_dir, self.save_name)

    def test_csmodel_reload_from_disk(self):
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            self.csmodel.save_path,
            cache_dir=os.path.join(self.save_dir, ".cache"),
            trust_remote_code=True
        )
        assert type(reloaded_model) == GPTNeoXForCausalLM
