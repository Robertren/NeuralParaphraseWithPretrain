# This script contains all the help function related to pre-trained word vector and deal with different data loader
from sklearn.feature_extraction import stop_words
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm

class ParaphraseData:
    """
    Class that represents a train/validation/test data
    """
    def __init__(self):
        pass
    # def __init__(self, raw_text, label, file_name):
    #     self.raw_text = raw_text
    #     self.label = label
    #     self.file_name = file_name
    #
    # def set_ngram(self, ngram_ctr):
    #     self.ngram = ngram_ctr
    #
    # def set_token_idx(self, token_idx):
    #     self.token_idx = token_idx
    #
    # def set_tokens(self, tokens):
    #     self.tokens = tokens


class ParaphraseDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    def __init__(self, data_dir):
        return

    def __len__(self):
        pass
        # return len(self.data['labels'])

    def __getitem__(self, key):
        pass
        # """
        # Triggered when you call dataset[i]
        # """
        # label = self.data['labels'][key]
        # token_idx = self.data['data_index'][key]
        # if len(token_idx) == 2:
        #     return ([], 0), label
        # else:
        #     token_idx = list(map(int, token_idx[1:-1].split(', ')))
        #     return (token_idx, len(token_idx)), label


def preprocess_text(text):
    """
    :param text: A sentence that has not been preprocessed
    :return: A preprocessed sentence
    """
    return


def extract_ngram_from_text(text, n, remove_stopwords=True):
    """
    :param text: A sentence which we want to extract character-level n_gram from
    :param n: n is the number of  character-level grams that you want to extract from text
    :param remove_stopwords: TODO: delete it or not
    :return:
    """
    return


def construct_ngram_indexer(ngram_counter_list, topk):
    """
    Function that selects the most common topk ngrams
    @param ngram_counter_list: list of counters
    @param topk: # of
    @return ngram2idx: a dictionary that maps ngram to an unique index
    """
    # find the top k ngram
    # maps the ngram to an unique index
    ngram_counter = Counter()
    for counter in tqdm(ngram_counter_list):
        ngram_counter.update(counter)
    ngram_counter_topk = ngram_counter.most_common(topk)
    ngram_indexer = {ngram_counter_topk[index][0]: index + 1 for index in range(len(ngram_counter_topk))}
    return ngram_indexer


def token_to_index(tokens, ngram_indexer):
    """
    Function that transform a list of tokens to a list of token index.
    @param tokens: list of ngram
    @param ngram_indexer: a dictionary that maps ngram to an unique index
    """
    # Please DO NOT assign any ngram to index 0 which is reserved for PAD token
    index_list = [ngram_indexer[token] for token in tokens if token in ngram_indexer]
    return index_list


def process_text_dataset(dataset, n, topk=None, ngram_indexer=None):
    """
    Top level function that encodes each datum into a list of ngram indices
    @param dataset: list of IMDBDatum
    @param n: n in "n-gram"
    @param topk: #
    @param ngram_indexer: a dictionary that maps ngram to an unique index
    """
    # extract n-gram
    for i in tqdm(range(len(dataset))):
        text_datum = dataset[i].raw_text
        ngrams, tokens = extract_ngram_from_text(text_datum, n)
        dataset[i].set_ngram(ngrams)
        dataset[i].set_tokens(tokens)
    # select top k ngram
    if ngram_indexer is None:
        ngram_indexer = construct_ngram_indexer([datum.ngram for datum in dataset], topk)
    # vectorize each datum
    for i in range(len(dataset)):
        dataset[i].set_token_idx(token_to_index(dataset[i].tokens, ngram_indexer))
    return dataset, ngram_indexer

# TODO: To decide if we need to use h5py format to save the data.