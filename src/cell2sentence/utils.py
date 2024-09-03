"""
Utility functions for cell2sentence
"""

#
# @authors: Rahul Dhodapkar, Syed Rizvi
#

import os
import sys
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
import plotnine as pn
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from datasets import Dataset, DatasetDict


# These words are reserved for data processing and utility functions
RESERVED_WORDS = [
    "cell_name",
    "cell_sentence",
]
BASE10_THRESHOLD = 3


def generate_vocabulary(adata):
    """
    Create a vocabulary dictionary, where each key represents a single gene
    token and the value represents the number of non-zero cells in the provided
    count matrix.

    Arguments:
        adata: an AnnData object to generate cell sentences from. Expects that
            `obs` correspond to cells and `vars` correspond to genes
    Return:
        a dictionary of gene vocabulary
    """
    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({})... "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    vocabulary = OrderedDict()
    gene_sums = np.ravel(np.sum(adata.X > 0, axis=0))

    for i, name in enumerate(adata.var_names):
        vocabulary[name.upper()] = gene_sums[i]  # keys are all uppercase gene names

    return vocabulary


def concat_vocabularies(vocabulary_list):
    """
    Helper function to concatenate multiple vocabulary ordered dictionaries.
    Preserves order of features in the first vocabulary, and appends any additional
    features from successive dictionaries.
    """
    concat_vocab = OrderedDict()
    for ordered_dict in vocabulary_list:
        for key, val in ordered_dict.items():
            if key not in concat_vocab:
                concat_vocab[key] = val
            else:
                concat_vocab[key] = concat_vocab[key] + val
    return concat_vocab


def generate_sentences(adata, vocab, delimiter=' ', random_state=42):
    """
    Transform expression matrix to sentences. Sentences contain gene "words"
    denoting genes with non-zero expression. Genes are ordered from highest
    expression to lowest expression.

    Arguments:
        adata: an AnnData object to generate cell sentences from. Expects that
            `obs` correspond to cells and `vars` correspond to genes
        vocab: an OrderedDict which as feature names as keys and counts as values
        random_state: sets the numpy random state for splitting ties
    
    Returns:
        a `numpy.ndarray` of sentences, split by delimiter.
    """
    np.random.seed(random_state)

    if len(adata.var) > len(adata.obs):
        print(
            (
                "WARN: more variables ({}) than observations ({}), "
                + "did you mean to transpose the object (e.g. adata.T)?"
            ).format(len(adata.var), len(adata.obs)),
            file=sys.stderr,
        )

    mat = sparse.csr_matrix(adata.X)
    enc_map = list(vocab.keys())

    sentences = []
    for i in tqdm(range(mat.shape[0])):
        # for row i, [indptr[i]:indptr[i+1]] returns the indices of elements to take from 
        #  data and indices corresponding to row i
        cols = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        vals = mat.data[mat.indptr[i] : mat.indptr[i + 1]]
        cols, vals = shuffle(cols, vals)
        sentences.append(delimiter.join([enc_map[x] for x in cols[np.argsort(-vals, kind="stable")]]))

    return sentences


def get_benchmark_df(
    normalized_expression: np.ndarray,
    rank_normalized_expression: np.ndarray,
    exclude_zeros: bool = True,
):
    """Build pandas DataFrame with normalized expression and ranks"""
    # filter out all zero-expression genes
    df = pd.DataFrame({"normalized_expression": np.ravel(normalized_expression)})
    if exclude_zeros:
        df = df[df["normalized_expression"] != 0]

    df["rank_normalized_expression"] = np.ravel(rank_normalized_expression)[df.index]

    # log-transform ranks and expression values
    df["log_rank_normalized_expression"] = np.log10(1 + df["rank_normalized_expression"])
    return df


def sort_transcript_counts(raw_data):
    """Sort transcript counts, yielding matrix of ranks"""
    rank_X = np.zeros(shape=raw_data.shape)
    for i in range(raw_data.shape[0]):
        cols = np.ravel(range(raw_data.shape[1]))
        vals = np.ravel(raw_data[i, :])
        cols, vals = shuffle(cols, vals)
        ranks = cols[np.argsort(-vals, kind="stable")]
        rank_X[i, ranks] = np.arange(len(ranks))

    return rank_X


def benchmark_expression_conversion(
    benchmark_output_dir: str,
    save_name: str,
    normalized_expression_matrix,
    sample_size: int = 1024,
):
    """
    Helper function to take a normalized counts matrix and compute rank transformation 
    and inverse transformation metrics. Saves plots and metrics to a subfolder called 
    save_name + '_benchmark'.

    Arguments:
        benchmark_output_dir: directory to store results to (subdirectory will be created)
        save_name: name of dataset being benchmarked
        normalized_expression_matrix:  numpy matrix of normalized counts
        sample_size: number of cells to sample for computing metrics and plots
    """
    # Create save directory
    benchmark_save_dir = os.path.join(benchmark_output_dir, save_name + "_benchmark")
    if not os.path.exists(benchmark_save_dir):
        os.mkdir(benchmark_save_dir)

    # Subsample a set of cells to evaluate on
    sample_size = min(sample_size, normalized_expression_matrix.shape[0])
    sample_idxs = np.random.choice(normalized_expression_matrix.shape[0], size=sample_size, replace=False)
    sample_idxs = np.sort(sample_idxs)
    normalized_expression = normalized_expression_matrix[sample_idxs, :].copy()  # copy to avoid mutating original obj
    normalized_expression = normalized_expression.todense()  # convert to dense array
    print(f"Benchmarking with a sample dataset of size {normalized_expression.shape[0]}")

    # Compute ranks from expression values
    rank_normalized_expression = sort_transcript_counts(normalized_expression)

    # Create benchmark DataFrame containing expression values and corresponding ranks
    benchmark_df = get_benchmark_df(
        normalized_expression,
        rank_normalized_expression,
    )
    
    # Setup for log rank vs log expression plot
    plot_config = {
        "title": "Normalized Expression vs Log Rank",
        "x": "log_rank_normalized_expression",
        "x_label": "Log Rank for Normalized Expression",
        "y": "normalized_expression",
        "y_label": "Normalized Gene Expression",
        "gt_x": "log_rank_normalized_expression",
    }

    x, y = plot_config["x"], plot_config["y"]
    x_val = benchmark_df.loc[benchmark_df[x] < BASE10_THRESHOLD, x].to_numpy().reshape(-1, 1)
    y_val = benchmark_df.loc[benchmark_df[x] < BASE10_THRESHOLD, y]

    # Plot the linear fit between log expression and log rank
    linear_reg = linear_model.LinearRegression().fit(x_val, y_val)
    plot = (
        pn.ggplot(benchmark_df, pn.aes(x=x, y=y))
        + pn.geom_abline(
            slope=linear_reg.coef_,
            intercept=linear_reg.intercept_,
            color="darkorange",
        )
        + pn.geom_point(color="blue", size=0.2)
        + pn.labs(
            x=plot_config["x_label"],
            y=plot_config["y_label"],
        )
    )
    output_filename = "_".join(plot_config["title"].lower().split(" ")) + ".png"
    output_filepath = os.path.join(benchmark_save_dir, output_filename)
    plot.save(output_filepath, dpi=300)

    # Reconstruct expression values from log rank (inverse transformation)
    reconstructed_expression = linear_reg.predict(
        benchmark_df[x].to_numpy().reshape(-1, 1)
    )

    # Calculate metrics of inverse transformation
    r_squared_score = metrics.r2_score(
        benchmark_df[y].values, reconstructed_expression
    )
    pearson_r_score = pearsonr(benchmark_df[y].values, reconstructed_expression)
    spearman_r_score = spearmanr(benchmark_df[y].values, reconstructed_expression)

    reconstructed_y_key = "Reconstructed Expression"
    reconstruction_df = pd.DataFrame({
        plot_config["y_label"]: benchmark_df[y],
        reconstructed_y_key: reconstructed_expression,
    })

    # Plot inverse transformation scatterplot
    plot = (
        pn.ggplot(
            reconstruction_df.sample(sample_size)
            if len(reconstruction_df) > sample_size
            else reconstruction_df,
            pn.aes(x=plot_config["y_label"], y=reconstructed_y_key),
        )
        + pn.geom_point(color="blue", size=0.2)
        + pn.geom_abline(slope=1, intercept=0, color="red")
        + pn.annotate(
            "text",
            x=1.5,
            y=0.75,
            label=f"R2: {r_squared_score:.2f}\nPearson: {pearson_r_score[0]:.2f}\nSpearman: {spearman_r_score[0]:.2f}",
            size=10,
            color="black",
            ha="left",
        )
        + pn.labs(
            x=plot_config.get("y_label"),
            y=reconstructed_y_key,
            title="Reconstruction of Normalized Expression from Rank",
        )
    )
    output_filename = reconstructed_y_key.lower().replace(" ", "_") + ".png"
    output_filepath = os.path.join(benchmark_save_dir, output_filename)
    plot.save(output_filepath, dpi=300)

    # Save DataFrame with metrics
    result_df = pd.DataFrame({
        "experiment": [plot_config["title"]],
        "x_axis": [x],
        "y_axis": [y],
        "threshold": [BASE10_THRESHOLD],
        "slope": [linear_reg.coef_.item()],
        "intercept": [linear_reg.intercept_.item()],
        "r_squared": [r_squared_score.item()],
        "pearson_r_statistic": [pearson_r_score.statistic.item()],
        "pearson_r_pvalue": [pearson_r_score.pvalue.item()],
        "spearman_r_statistic": [spearman_r_score.statistic.item()],
        "spearman_r_pvalue": [spearman_r_score.pvalue.item()],
    })

    # save benchmarking results
    benchmark_results_filepath = os.path.join(benchmark_save_dir, "c2s_transformation_metrics.csv")
    result_df.to_csv(benchmark_results_filepath, index=False)


def build_arrow_dataset(
    cell_names: list, 
    sentences: list, 
    adata, 
    label_col_names: list
):
    """
    Build an arrow dataset from a list of cell IDs and cell sentences. Optionally
    include columns for additional cell metadata.

    Arguments:
        cell_names: list of strings representing (unique) cell identifiers
        sentences: list of strings representing cell sentences
        adata: anndata.AnnData object
        label_col_names: list of column names in .obs DataFrame to save into dataset
                                along with cell sentences
    
    Returns:
        Arrow dataset
    """
    if label_col_names is not None:
        for word in RESERVED_WORDS:
            assert word not in label_col_names, f"Reserved keyword {word} found in label column names."
    
    # Build dataset dictionary
    data_dict = {
        "cell_name": cell_names,
        "cell_sentence": sentences,  # key names stored once in metadata file
    }
    
    if label_col_names is not None:
        for label_col in label_col_names:
            data_dict[label_col] = adata.obs[label_col].tolist()

    # Create arrow dataset
    full_ds = Dataset.from_dict(data_dict)
    return full_ds


def train_test_split_arrow_ds(arrow_ds):
    """
    Helper function to split an arrow dataset into train, val, and test sets with
    a 80/10/10 split ratio.

    Arguments:
        arrow_ds: arrow dataset to split
    
    Returns:
        Tuple of i) dataset dictionary with train, val, and test splits, and 
        ii) dictionary of indices of cells in each split
    """
    # Train/val/test split
    cell_indices_list = list(range(arrow_ds.num_rows))
    train_and_val_indices, test_indices = train_test_split(cell_indices_list, test_size=0.1)
    train_indices, val_indices = train_test_split(train_and_val_indices, test_size=0.11)

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()
    data_split_indices_dict = { "train": train_indices, "val": val_indices, "test": test_indices }

    train_ds = arrow_ds.select(data_split_indices_dict["train"])
    val_ds = arrow_ds.select(data_split_indices_dict["val"])
    test_ds = arrow_ds.select(data_split_indices_dict["test"])

    # Create the DatasetDict with train, validation, and test splits
    ds_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })
    return ds_dict, data_split_indices_dict


def tokenize_loss_on_response(examples, tokenizer, ignore_token_id: int = -100):
    """Tokenize LLM input + response, loss taken only on model response."""
    prompt_inputs = examples["model_input"]
    responses = examples["response"]
    model_inputs = tokenizer(prompt_inputs)
    labels = tokenizer(responses)

    for i in range(len(prompt_inputs)):
        prompt_input_ids = model_inputs["input_ids"][i]
        response_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        model_inputs["input_ids"][i] = prompt_input_ids + response_input_ids
        labels["input_ids"][i] = [ignore_token_id] * len(prompt_input_ids) + response_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_all(examples, tokenizer):
    """Tokenize LLM input + response, loss taken on all tokens."""
    # Build list of full input prompts: model_input + response, separated by a space
    full_inputs = []
    num_samples = len(examples["model_input"])
    for sample_idx in range(num_samples):
        full_inputs.append(examples["model_input"][sample_idx] + " " + examples["response"][sample_idx])

    # Tokenize full input prompts using LLM tokenizer -> will turn prompts into input indices for LLM
    model_inputs = tokenizer(full_inputs)
    for i in range(num_samples):
        model_inputs["input_ids"][i] += [tokenizer.eos_token_id]
        model_inputs["attention_mask"][i] += [tokenizer.eos_token_id]
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs


def post_process_generated_cell_sentences(
    cell_sentence: str,
    vocab_list: list,
    replace_nonsense_string: str = "NOT_A_GENE",
):
    """
    Helper function to replace non-gene words in generated sentences and deal with
    duplicated genes by averaging their positions in the sentence.

    Arguments:
        cell_sentence_str: string representing a cell sentence
        vocab_list: list of all gene feature names, expression vector will be ordered
            following this list
        replace_nonsense_string: word to replace non-gene words with (warning will be removed
            from generated cell sentences, do not choose a gene name)
    
    Returns:
        Tuple of i) post processed cell sentence gene list and ii) number of non-genes replaced
    """
    # Convert the cell sentence to uppercase and split into words
    words = cell_sentence.upper().split(" ")
    # Replace words not in the vocabulary with the replace_nonsense_string
    generated_gene_names = [word if word in vocab_list else replace_nonsense_string for word in words]
    num_genes_replaced = generated_gene_names.count(replace_nonsense_string)

    # Calculate average ranks
    gene_name_to_occurrences = Counter(generated_gene_names)  # maps gene name --> # of occurrences
    post_processed_sentence = generated_gene_names.copy()

    for gene_name in gene_name_to_occurrences:
        if (gene_name_to_occurrences[gene_name] > 1 and gene_name != replace_nonsense_string):
            # Find positions of all occurrences of duplicated generated gene in list
            # Note: using post_processed_sentence here; since duplicates are being removed, list will be
            #   getting shorter. Getting indices in original list will no longer be accurate positions
            occurrence_positions = [idx for idx, elem in enumerate(post_processed_sentence) if elem == gene_name]
            average_position = int(sum(occurrence_positions) / len(occurrence_positions))

            # Remove occurrences
            post_processed_sentence = [elem for elem in post_processed_sentence if elem != gene_name]
            # Reinsert gene_name at average position
            post_processed_sentence.insert(average_position, gene_name)
    
    return post_processed_sentence, num_genes_replaced


def reconstruct_expression_from_cell_sentence(
    cell_sentence_str: str,
    delimiter: str,
    vocab_list: list,
    slope: float,
    intercept: float,
):
    """
    Helper function to reconstruct an expression vector from a cell sentence.

    Arguments:
        cell_sentence_str: string representing a cell sentence
        delimiter: character which separates gene names in the cell sentence
        vocab_list: list of all gene feature names, expression vector will be ordered
            following this list
        slope: slope of linear model fit on log rank vs log expression
        intercept: intercept of linear model fit on log rank vs log expression
    
    Returns:
        Expression vector numpy array
    """
    # Split cell sentence string into list of gene "words"
    cell_sentence = cell_sentence_str.split(delimiter)

    # Create a mapping from gene names to their vocab indices for O(1) lookups
    gene_to_index = {gene: idx for idx, gene in enumerate(vocab_list)}

    # Initialize the expression vector with zeros
    expression_vector = np.zeros(len(vocab_list), dtype=np.float32)

    # Pre-compute the log rank values for all positions in cell sentence
    log_ranks = np.log10(1 + np.arange(len(cell_sentence)))

    # Calculate gene expression values and update the expression vector
    for pos, gene_name in enumerate(cell_sentence):
        gene_idx_in_vector = gene_to_index.get(gene_name)
        if gene_idx_in_vector is not None:  # gene is in vocab_list
            gene_expr_val = intercept + (slope * log_ranks[pos])
            expression_vector[gene_idx_in_vector] = gene_expr_val

    return expression_vector
