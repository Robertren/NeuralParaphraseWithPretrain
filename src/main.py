import numpy
import torch
def calculate_aligned_phrases(dot_products, phrases_a, phrases_b, biases, self_attention = False):
    """
    :param dot_products: A list of dot products that came from first attend feed forward network
    :param phrases: The phrases of a single word, NOTE: A sentence has been decomposed to a list of phrases
    :param biases: The bias for computing learned distance-sensitive bias term
    :return: sub phrase in b (or a) that is softly aligned to a_i or (b_i)
    """
    if not self_attention:
        exp = torch.exp(dot_products)
        sum_beta = torch.sum(exp, 1)
        sum_alpha = torch.sum(exp, 0)
        div1 = exp / sum_beta.view(-1,1)
        div2 = exp / sum_alpha.view(1,-1)
        alpha, beta = torch.matmul(div2.transpose(0,1), phrases_a), torch.matmul(div1, phrases_b)
    # else:
    return alpha, beta


def concat_representation(phrases, aligned_phrase):
    """
    :param phrases: The original phrase
    :param aligned_phrase: The aligned_phrase that attach to the origin
    :return: A concatenation representation
    """
    return torch.cat((phrases, aligned_phrase), 1)



