CSModel
=======

A CSModel object is a wrapper around a Cell2Sentence model, which tracks the path of the model 
saved on disk. When needed, the model is loaded from the path on disk for inference or finetuning.
The class contains utilities for model generation and cell embedding with a Huggingface backend.

.. autofunction:: csmodel.CSModel

.. autofunction:: csmodel.CSModel.__init__

.. autofunction:: csmodel.CSModel.__str__

.. autofunction:: csmodel.CSModel.fine_tune

.. autofunction:: csmodel.CSModel.generate_from_prompt

.. autofunction:: csmodel.CSModel.generate_from_prompt_batched

.. autofunction:: csmodel.CSModel.embed_cell

.. autofunction:: csmodel.CSModel.embed_cells_batched

.. autofunction:: csmodel.CSModel.push_model_to_hub
