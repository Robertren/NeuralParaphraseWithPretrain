import torch
import torch.nn as nn


class AttendForwardNet(nn.module):
    """
    First Attend Feed Forward Network to calculate e_ij
    """
    def __init__(self):
        super(AttendForwardNet, self).__init__()
        pass

    def forward(self):
        return


class SelfAttentionForwardNet(nn.module):
    """
    The model which help calculate self attention to encode relationships between words with each sentence
    """
    def __init__(self):
        super(SelfAttentionForwardNet, self).__init__()
        pass

    def forward(self):
        return


class SelfAttentionForwardNetAligned(nn.module):
    """
    TODO: Discuss if we need to delete this model
    This model can be the same with the above one
    The model which help calculate self attention to encode relationships between words with each sentence
    """

    def __init__(self):
        super(SelfAttentionForwardNetAligned, self).__init__()
        pass

    def forward(self):
        return


class CompareForwardNet(nn.module):
    """
    This is the feed forward network to compare aligned phrases
    """

    def __init__(self):
        super(CompareForwardNet, self).__init__()
        pass

    def forward(self):
        return


class AggregateForwardNet(nn.module):
        """
        This is the feed forward network to aggregate all the outputs from the compare network and
        give a prediction of the label.
        """

        def __init__(self):
            super(AggregateForwardNet, self).__init__()
            pass

        def forward(self):
            return
