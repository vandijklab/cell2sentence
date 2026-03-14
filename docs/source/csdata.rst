CSData
======

A CSData object is a wrapper around a single-cell dataset, which tracks its path on disk.
It's main functionality is to keep the dataset stored on disk so that data need not be 
loaded in memory until it is used, either for inference or finetuning.

.. autofunction:: csdata.CSData

.. autofunction:: csdata.CSData.__init__


The ``CSData.adata_to_arrow()`` function takes as input a single-cell dataset in the form 
of a H5AD Scanpy object, and returns a pyarrow dataset which stores cell sentences 
representing the dataset after C2S rank transformation.

.. autofunction:: csdata.CSData.adata_to_arrow

.. autofunction:: csdata.CSData.csdata_from_arrow

.. autofunction:: csdata.CSData.csdata_from_multiple_arrow_datasets

.. autofunction:: csdata.CSData.get_sentence_strings

.. autofunction:: csdata.CSData.__str__
