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
from peft import PeftModel
# Local imports
import cell2sentence as cs
from cell2sentence.csmodel import CSModel

HERE = Path(__file__).parent


# class TestCSModelCellTypeConditionalGenerationWorkflow:
#     @classmethod
#     def setup_class(self):
#         # Define CSModel object
#         cell_type_cond_generation_model_path = "/home/sr2464/scratch/C2S_Files/multicell_pretraining_v2_important_models/pythia-410m-multicell_v2_2024-07-28_14-10-44_checkpoint-7000_cell_type_cond_generation"
#         self.save_dir = "/home/sr2464/scratch/C2S_Files/c2s_api_testing/csmodel_testing"
#         self.save_name = "cell_type_cond_generation_pythia_410M_1"
#         self.csmodel = CSModel(
#             model_name_or_path=cell_type_cond_generation_model_path,
#             save_dir=self.save_dir,
#             save_name=self.save_name
#         )

#     def test_csmodel_string_representation(self):
#         assert 'CSModel' in (str(self.csmodel) + '')

#     def test_csmodel_created_correctly(self):
#         assert self.csmodel.save_path == os.path.join(self.save_dir, self.save_name)

#     def test_csmodel_reload_from_disk(self):
#         reloaded_model = AutoModelForCausalLM.from_pretrained(
#             self.csmodel.save_path,
#             cache_dir=os.path.join(self.save_dir, ".cache"),
#             trust_remote_code=True
#         )
#         assert type(reloaded_model) == GPTNeoXForCausalLM

class TestCSModelPeftModelLoadingAndErrorHandling:
    @classmethod
    def setup_class(self):
        self.save_dir = "/mnt/c/Users/khmam/Desktop/c2s_model_directory"
        self.save_name = "lora_gemma_model"
        hf_model_path = "vandijklab/C2S-Scale-Gemma-2-2B"
        self.csmodel = CSModel(
            model_name_or_path=hf_model_path,
            save_dir=self.save_dir,
            save_name=self.save_name,
            peft = True,
        )

    def test_csmodel_created_correctly(self):
        assert self.csmodel.save_path == os.path.join(self.save_dir, self.save_name)
    
    def test_layers_are_created_correctly(self):
        from peft import PeftModel, AutoPeftModelForCausalLM
        
        # Load the model back from disk using PEFT's loading method
        loaded_model = AutoPeftModelForCausalLM.from_pretrained(
            self.csmodel.save_path,
            trust_remote_code=True,
            is_trainable=False
        )
        
        # Verify it loaded as a PEFT model
        assert isinstance(loaded_model, PeftModel), "Model is not a PeftModel"
        
        # Verify that LoRA layers are present in the loaded model
        lora_modules = [name for name, module in loaded_model.named_modules() if "lora" in name.lower()]
        assert len(lora_modules) > 0, "No LoRA layers found in the reloaded model modules"
        
        # Ensure the active adapter is set (typical for LoRA)
        assert hasattr(loaded_model, "active_adapter"), "No active adapter found on the PEFT model"
