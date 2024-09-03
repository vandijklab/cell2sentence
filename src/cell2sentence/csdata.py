"""
Main data wrapper class definition
"""

#
# @authors: Rahul Dhodapkar, Syed Rizvi
#

# Python built-in libraries
import os

# Third-party libraries
from datasets import load_from_disk, concatenate_datasets

# Local imports
from cell2sentence.utils import (
    generate_vocabulary, 
    concat_vocabularies,
    generate_sentences, 
    build_arrow_dataset
)


class CSData():
    """
    Wrapper class to abstract different types of input data that can be passed
    in cell2sentence based workflows.
    """

    def __init__(self, vocab, data_path, dataset_backend='arrow'):
        """
        Core constructor for CSData object that contains a data path to the C2S dataset.
        Helps manage data loading and buffering.

        Arguments:
            vocab: ordered dictionary of feature names and their number of cells expressed
            data_path: path to saved arrow dataset on disk
            dataset_backend: backend implementation of C2S dataset (currently Huggingface supported)
        """
        self.vocab = vocab  # Ordered Dictionary: {gene_name: num_expressed_cells}
        self.data_path = data_path  # path to data file in arrow format
        self.dataset_backend = dataset_backend  # support plaintext and arrow

    @classmethod
    def adata_to_arrow(self, 
        adata, 
        random_state: int = 42, 
        sentence_delimiter: str = ' ',
        label_col_names: list = None, 
    ):
        """
        Construct an arrow dataset of cell sentences from an AnnData object.

        Arguments:
            adata: anndata.AnnData object to convert into a cell sentence dataset
            random_state: random seed to control randomness
            sentence_delimiter: separator for cell sentence strings (default: ' ')
            label_col_names: optional list of column names in .obs to save into dataset along with cell sentences
        
        Returns:
            Tuple of i) arrow dataset of cell sentences and ii) ordered dictionary of gene names and their number of expressed cells
        """
        # Warn if var_names contains ensembl IDs instead of gene names.
        first_gene_name = str(adata.var_names[0])
        if "ENS" in first_gene_name:
            print(
                """WARN: adata.var_names seems to contain ensembl IDs rather than gene/feature names. 
                It is highly recommended to use gene names in cell sentences."""
            )

        # Create vocabulary and cell sentences based on adata object
        vocabulary = generate_vocabulary(adata)
        sentences = generate_sentences(adata, vocabulary, delimiter=sentence_delimiter)
        cell_names = adata.obs_names.tolist()

        # Turn into arrow dataset
        arrow_dataset = build_arrow_dataset(
            cell_names=cell_names, 
            sentences=sentences,
            adata=adata,
            label_col_names=label_col_names
        )
        return arrow_dataset, vocabulary

    @classmethod
    def csdata_from_arrow(self, 
        arrow_dataset, 
        vocabulary,
        save_dir: str, 
        save_name: str,
        dataset_backend: str = 'arrow',
    ):
        """
        Create new CSData object from a single arrow dataset.

        This function expects the C2S arrow dataset to already be created (e.g. by using
        adata_to_arrow()), and will create a CSData() wrapper object which will manage
        loading the dataset in concert with other C2S classes.

        Arguments:
            arrow_dataset: an arrow dataset to create the CSData wrapper around
            vocabulary: ordered dictionary containing feature names and their number of cells expressed
            save_dir: directory where cell sentence dataset will be saved to disk
            save_name: name of folder to create storing cell sentence dataset (will be created)
            dataset_backend: backend implementation for cell sentence dataset
        
        Returns:
            CSData() object
        """
        assert dataset_backend in ['arrow'], "C2S currently only supports arrow backend."
        
        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        
        # Save to disk
        arrow_dataset.save_to_disk(save_path)

        return self(
            vocab=vocabulary,
            data_path=save_path,
            dataset_backend='arrow',
        )
    
    @classmethod
    def csdata_from_multiple_arrow_datasets(self, 
        arrow_dataset_list, 
        vocabulary_list,
        save_dir: str, 
        save_name: str,
        dataset_backend: str = 'arrow',
    ):
        """
        Create new CSData object from multiple arrow datasets. Useful when creating a CSData()
        object to manage multiple chunks of a large single-cell dataset, or when combining
        multiple single-cell datasets into one large dataset.

        Arguments:
            arrow_dataset_list: list of arrow datasets to create the CSData object around
            vocabulary_list: list of ordered dictionaries containing feature names and their number of cells expressed
            save_dir: directory where cell sentence dataset will be saved to disk
            save_name: name of folder to create storing cell sentence dataset (will be created)
            dataset_backend: backend implementation for cell sentence dataset
        
        Returns:
            CSData() object
        """
        assert dataset_backend in ['arrow'], "C2S currently only supports arrow backend."
        
        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        
        # Concatenate datasets
        arrow_dataset = concatenate_datasets(arrow_dataset_list)
        vocabulary = concat_vocabularies(vocabulary_list)

        # Save to disk
        arrow_dataset.save_to_disk(save_path)

        return self(
            vocab=vocabulary,
            data_path=save_path,
            dataset_backend='arrow',
        )

    def get_sentence_strings(self):
        """
        Helper function to return cell sentences sotred in arrow dataset.
        """
        arrow_ds = load_from_disk(self.data_path)
        return arrow_ds["cell_sentence"]

    def __str__(self):
        """
        Summarize CSData object as string for debugging and logging.
        """
        return f"CSData Object; Path={self.data_path}, Format={self.dataset_backend}"
