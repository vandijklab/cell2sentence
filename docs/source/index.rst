.. Cell2Sentence documentation master file, created by
   sphinx-quickstart on Mon Sep  2 12:15:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cell2Sentence: Single-cell Analysis With LLMs
=============================================

**Cell2Sentence (C2S)** is a framework for directly adapting Large Language Models (LLMs) to single-cell biology. C2S proposes a rank-ordering transformation of cell expression into cell sentences, which are sentences of space-separated gene names ordered by descending expression. By representing single-cell data as cell sentences, C2S provides a framework for LLMs to *directly* model single-cell biology in natural language, enabling diverse capabilities on multiple single-cell tasks.

C2S is developed by members of the `vanDijk Lab <https://www.vandijklab.org/>`_ at Yale University. Check out the :doc:`quickstart` section for quickstart instructions.

.. note::

   We are actively adding more features and documentation to the C2S API. For any feature requests or issues, please leave a GitHub issue or reach out to us!


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quickstart
   tasks
   csdata
   csmodel
   prompt_formatter
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
