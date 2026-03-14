# Cell2Sentence: Single-cell Analysis With LLMs

[![Code License: Apache 2.0](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://badge.fury.io/py/cell2sentence.svg)](https://badge.fury.io/py/cell2sentence)
[![DOI:10.1101/2025.04.14.648850](http://img.shields.io/badge/DOI-10.1101/2025.04.14.648850-B31B1B.svg)](https://doi.org/10.1101/2025.04.14.648850)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310020/)

![cell2sentence workflow image](c2s_overview_figure.png)

Implementation of [**"Scaling Large Language Models for Next-Generation Single-Cell Analysis"**](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v2), the latest Cell2Sentence (C2S-Scale) framework for applying Large Language Models (LLMs) to single-cell transcriptomics.  The previous paper establishing the C2S methodology can be found [here](https://www.biorxiv.org/content/10.1101/2023.09.11.557287v4).
C2S proposes a rank-ordering transformation of expression vectors into *cell sentences*—space-separated gene names ordered by descending expression—which allows LLMs to natively model scRNA-seq data using natural language.

The new **C2S-Scale** models scale to 27B parameters, unify transcriptomic and textual data, and enable advanced single-cell tasks including perturbation prediction, dataset summarization, and cluster captioning, and biological question answering. C2S-Scale models based on the Pythia architecture are already available on [Huggingface](https://huggingface.co/collections/vandijklab/cell2sentence-models-66d71f690a7b77558a36b9ef), and models based on Gemma-2 will soon be made available.

For more information, check out the [paper](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v2), explore the [documentation](https://vandijklab-cell2sentence.readthedocs.io/), or reach out to us at the [van Dijk Lab](https://www.vandijklab.org/)!



## News
🚀 (10/15/2025) The **C2S-Scale Gemma-2 2B and 27B** pretrained models are now available on HuggingFace: [C2S-Scale-Gemma-Models](https://huggingface.co/collections/vandijklab/c2s-scale-gemma-models-68ed5e4d3b55c8c29682d842). Check out what's new in our [updated preprint](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v2) and [blog post](https://blog.google/technology/ai/google-gemma-ai-cancer-therapy-discovery/) from Google!

📰 (04/17/2025) We published a **blog post** in collaboration with Google Research and Google DeepMind! Read it [here](https://research.google/blog/teaching-machines-the-language-of-biology-scaling-large-language-models-for-next-generation-single-cell-analysis/) to learn how Cell2Sentence is enabling next-gen single-cell discovery with LLMs.

🧠 (04/17/2025) We released the new **C2S-Scale preprint**, including larger models and tons of new experiments! Read the full paper on bioRxiv [here](https://www.biorxiv.org/content/10.1101/2025.04.14.648850v2).

🚀 (04/17/2025) The **C2S-Scale-1B** pretrained model is now available on HuggingFace: [vandijklab/c2s-scale-1b](https://huggingface.co/vandijklab/C2S-Scale-Pythia-1b-pt).

🎉 (12/08/2024) Added built-in support for finetuning on your own custom prompt templates as well as multi-cell prompt formatting! See tutorials 7 & 8 for examples.

🎉 (09/03/2024) We release the new C2S code base, with core functionalities for working with C2S models in common single-cell workflows!

🎉 (09/03/2024) We release a new suite of Pythia models for cell type prediction, cell type conditioned generation, and a diverse multi-cell multi-task model! These models are trained on over 57 million human and mouse cells from CellxGene and Human Cell Atlas.

🎉 (05/01/2024) Cell2Sentence was accepted to ICML 2024! Congratulations to all of the authors involved in the project. See you in Vienna!

🎉 (02/15/2024) **pythia-160m-c2s** trained on full cell sentences is available on the Hugging Face hub [here](https://huggingface.co/vandijklab/pythia-160m-c2s)! This new model generates and predicts cell types from entire cells directly in text with the [Pythia-160 base model](https://huggingface.co/EleutherAI/pythia-160m).

🎉 (02/15/2024) Our **updated preprint** was posted on BioRxiv [here](https://www.biorxiv.org/content/10.1101/2023.09.11.557287v3). We introduce our latest results, including full cell sentence generation, combinatorial cell label prediction, abstract generation, and training on a large multi-tissue dataset of 36M cells.


## Installation

To set up a cell2sentence environment, first pull the repository locally:
```bash
git clone https://github.com/vandijklab/cell2sentence.git
```

Navigate a terminal into the root of the repository. Next, create an Anaconda environment using `python3` using [anaconda](https://docs.anaconda.com/anaconda/install/) with:
```bash
conda create -n cell2sentence python=3.10
```

Next, activate the environment:
```bash
conda activate cell2sentence
```

Finally, run the setup:
```bash
make install
```

This will install the latest development environment of cell2sentence, along with other pacakge dependendies. You can also install cell2sentence itself using `pip`:
```bash
pip install cell2sentence==1.2.0
```

The C2S package will allow usage of the core functionalities of C2S, including inference using existing C2S models and finetuning your own C2S models on your own datasets.

If you would like to speed up inference, you can optionally install flash-attention, which speeds up inference times particularly for long sequences (e.g. for generating cells with more than a few hundred genes):
```bash
pip install flash-attn --no-build-isolation
```
Detailed instructions for installing flash-attention can be found in the official [installation instructions](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features). To enable flash attention 2, see the example in tutorial notebook 5 (cell generation).

## Tutorials

The following notebooks provide guides on common workflows with C2S models. For more information about specific functions and modules, check out the [official C2S documentation page](https://vandijklab-cell2sentence.readthedocs.io/).

| Notebook | Description                                             |
----------|---------------------------------------------------------|
| [c2s_tutorial_0_data_preparation.ipynb](tutorials/c2s_tutorial_0_data_preparation.ipynb) | Data loading and preprocessing example on an immune tissue dataset
| [c2s_tutorial_1_rank_transformation_and_reconstruction.ipynb](tutorials/c2s_tutorial_1_rank_transformation_and_reconstruction.ipynb) | C2S rank transformation to cell sentences and inverse transformation
| [c2s_tutorial_2_cell_embedding.ipynb](tutorials/c2s_tutorial_2_cell_embedding.ipynb) | Obtaining cell type embeddings with C2S models
| [c2s_tutorial_3_finetuning_on_new_datasets.ipynb](tutorials/c2s_tutorial_3_finetuning_on_new_datasets.ipynb) | Finetuning C2S models on new datasets
| [c2s_tutorial_4_cell_type_prediction.ipynb](tutorials/c2s_tutorial_4_cell_type_prediction.ipynb) | Cell type prediction using C2S models
| [c2s_tutorial_5_cell_generation.ipynb](tutorials/c2s_tutorial_5_cell_generation.ipynb) | Cell generation conditioned on cell type
| [c2s_tutorial_6_cell_annotation_with_foundation_model.ipynb](tutorials/c2s_tutorial_6_cell_annotation_with_foundation_model.ipynb) | Cell type annotation with foundation model
| [c2s_tutorial_7_custom_prompt_templates.ipynb](tutorials/c2s_tutorials_7_custom_prompt_templates.ipynb) | Custom Prompt Templates with C2S PromptFormatter class
| [c2s_tutorial_8_multi_cell_tissue_prediction.ipynb](tutorials/c2s_tutorial_8_multi_cell_tissue_prediction.ipynb) | Classifying the Tissue based on Multiple cell sentences
| [c2s_tutorial_9_natural_language_interpretation.ipynb](tutorials/c2s_tutorial_9_natural_language_interpretation.ipynb) | Use the C2S model to generate insightful summaries for different sets of cells
| [c2s_tutorial_10_perturbation_response_prediction.ipynb](tutorials/c2s_tutorial_10_perturbation_response_prediction.ipynb)|


## Model Zoo

The following is a collection of C2S pretrained models which are released for open use. Each is available through Huggingface, and has different capabilities
and pretraining recipies. More details on each model are provided in their respective Huggingface cards, and tutorial notebooks for different C2S workflows
each explain which model they use.

| Model name | Task(s)                   | Training Data                         |
----------|---------------------------------------------------------|------|
| [C2S-Pythia-410M cell type prediction model](https://huggingface.co/vandijklab/C2S-Pythia-410m-cell-type-prediction) | Next gene prediction, cell type prediction | CellxGene, HCA, cell type annotations
| [C2S-Pythia-410M cell type conditioned single cell generation model](https://huggingface.co/vandijklab/C2S-Pythia-410m-cell-type-conditioned-cell-generation) | Next gene prediction, cell type conditioned single-cell generation | CellxGene, HCA, cell type annotations
| [C2S-Pythia-410M diverse single- and multi-cell tasks model](https://huggingface.co/vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks) | Next gene prediction, diverse single- and multi-cell prediction and generation tasks | CellxGene, HCA, cell type and tissue annotations, paper abstracts
| [C2S-Pythia-160M full cell generation (legacy)](https://huggingface.co/vandijklab/pythia-160m-c2s) | Next gene prediction, cell type prediction | Immune tissue ([Domínguez Conde et al.](https://www.science.org/doi/full/10.1126/science.abl5197)), cell type annotations
| GPT-2 Large foundation model (legacy) | Next gene prediction, cell type prediction, conditional single-cell generation | CellxGene, HCA, cell type and tissue annotations, paper abstracts


## Feature Development
- [x] Add core functionality for cell sentence conversion, data manipulation
- [x] Add CSModel wrapper, finetuning functionality
- [x] Add prompt formatting, support for multiple tasks
- [x] Add functionality for inverse reconstruction back to expression vectors
- [x] Add tutorial notebooks for main C2S workflows: cell type prediction, cell generation
- [x] Add multi-cell prompt formatting
- [ ] Add support for legacy C2S-GPT-2 model prompts
- [x] Add parameter-efficient finetuning methods (LoRA)


## License
Licensed under the [Apache License 2.0](LICENSE).  


## Cite Cell2Sentence

If you use Cell2Sentence in your work, please cite our paper:

```bibtex
@article{rizvi2025scaling,
  title={Scaling large language models for next-generation single-cell analysis},
  author={Rizvi, Syed Asad and Levine, Daniel and Patel, Aakash and Zhang, Shiyang and Wang, Eric and Perry, Curtis Jamison and Constante, Nicole Mayerli and He, Sizhuang and Zhang, David and Tang, Cerise and others},
  journal={BioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
