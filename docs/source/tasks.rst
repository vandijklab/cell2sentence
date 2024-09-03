Tasks
=====

Cell Generation
---------------

To generate cells conditioned on cell type using a C2S model,
you can use the ``tasks.generate_cells_conditioned_on_cell_type()`` function.
This function will call the batched generation function of the CSModel class
with cell type generation prompts.

.. autofunction:: tasks.generate_cells_conditioned_on_cell_type


Cell Type Annotation
--------------------

To predict cell types of data, you can use the 
``tasks.predict_cell_types_of_data()`` function:

.. autofunction:: tasks.predict_cell_types_of_data


Cell Embedding
--------------

To embed cells using C2S models, you can use the 
``tasks.embed_cells()`` function. This function loads a CSModel object, and
uses the C2S model to embed cell sentences from the CSData object into 
embedding vectors.

.. autofunction:: tasks.embed_cells

