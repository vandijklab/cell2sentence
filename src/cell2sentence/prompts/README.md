# Prompts ReadME
This text file documents the structure of Cell2Sentence (C2S) prompts, which are located in JSON files
within this directory.

## Details
1. Each JSON file in this directory corresponds to a task which we want to perform using C2S, for example, single cell generation conditioned on a cell's cell type label.
2. Within a JSON file (e.g. single_cell_cell_type_conditional_generation>prompts), there are two keys: (i) 'model_input' and 'response'.
    - 'model_input' is the input prompt for the LLM, in other words, the sentence that goes into the LLM asking it to perform a certain task.
    - 'response' is the answer which the C2S model is supposed to generate in response to the input prompt.
3. There are certain keys, enclosed with braces ('{' and '}'). These keys will be replaced by a prompt formatter object with values depending on the task. For example:
    - If we are using the top 100 genes for cell type classification, then '{num_genes}' will be replaces with '100'.
    - For a human cell, '{organism}' would be replaces with 'Homo sapiens'.
    - '{cell_type}' would be replaced with the cell type label of the cell.
    - '{single_cell_sentence}' would be replaced with the cell sentence representing that cell.
4. In each JSON file, there are multiple variations of the prompt. This is to provide diversity of language in how we are asking C2S to perform given tasks.

# Overall workflow
1. First, given the task which we are performing with C2S, we create a prompt formatter object that loads the correct prompt JSON file.
2. For a given cell which we input into the LLM, we sample one of the prompt variations.
3. We replace the keys ('{...}') with their correct values based on the cell sentence, metadata, and task
4. The formatted prompt is then ready to be input into the LLM, either in a training or inference script/notebook.
