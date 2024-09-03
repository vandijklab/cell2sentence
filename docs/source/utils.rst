Utils
=====

This module contains utility functions for various C2S-related workflows, such as 
rank transformation, train/test splitting, and tokenization during finetuning.


.. autofunction:: utils.generate_vocabulary

.. autofunction:: utils.concat_vocabularies

.. autofunction:: utils.generate_sentences

.. autofunction:: utils.get_benchmark_df

.. autofunction:: utils.sort_transcript_counts

.. autofunction:: utils.benchmark_expression_conversion

.. autofunction:: utils.build_arrow_dataset

.. autofunction:: utils.train_test_split_arrow_ds

.. autofunction:: utils.tokenize_loss_on_response

.. autofunction:: utils.tokenize_all

.. autofunction:: utils.post_process_generated_cell_sentences

.. autofunction:: utils.reconstruct_expression_from_cell_sentence

