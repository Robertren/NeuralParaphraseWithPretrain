import torch
import torch.nn as nn


class AttendForwardNet(nn.Module):
    """
    First Attend Feed Forward Network to calculate e_ij
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, p):
        super(AttendForwardNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_size, padding_idx = 0)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)


    def forward(self, index):
        data = self.embedding(index)
        data = self.dropout(self.linear1(data))
        data = self.relu(data)
        data = self.dropout(self.linear2(data))
        data = self.relu(data)
        data = data.sum(0)
        return data



class SelfAttentionForwardNet(nn.Module):
    """
    The model which help calculate self attention to encode relationships between words with each sentence
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, p):
        super(SelfAttentionForwardNet, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, data):
        data = self.dropout(self.linear1(data))
        data = self.relu(data)
        data = self.dropout(self.linear2(data))
        data = self.relu(data)
        data = data.sum(0)
        return data


class SelfAttentionForwardNetAligned(nn.Module):
    """
    TODO: Discuss if we need to delete this model
    This model can be the same with the above one
    The model which help calculate self attention to encode relationships between words with each sentence
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, p):
        super(SelfAttentionForwardNetAligned, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, data):
        data = self.dropout(self.linear1(data))
        data = self.relu(data)
        data = self.dropout(self.linear2(data))
        data = self.relu(data)
        data = data.sum(0)
        return data


class CompareForwardNet(nn.Module):
    """
    This is the feed forward network to compare aligned phrases
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, p):
        super(CompareForwardNet, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, data):
        data = self.dropout(self.linear1(data))
        data = self.relu(data)
        data = self.dropout(self.linear2(data))
        data = self.relu(data)
        data = data.sum(0)
        return data


class AggregateForwardNet(nn.Module):
        """
        This is the feed forward network to aggregate all the outputs from the compare network and
        give a prediction of the label.
        """

        def __init__(self, vocab_size, embedding_size, hidden_size1, hidden_size2, output_size, p=0.1):
            super(AggregateForwardNet, self).__init__()
            self.linear1 = nn.Linear(embedding_size, hidden_size1)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_size1, hidden_size2)
            self.linear3 = nn.Linear(hidden_size2, output_size)
            self.dropout = nn.Dropout(p)
            self.softmax = nn.LogSoftmax()

        def forward(self, data):
            # dropout 0.1
            data = self.dropout(self.linear1(data))
            data = self.relu(data)
            data = self.dropout(self.linear2(data))
            data = self.relu(data)
            output = self.softmax(self.linear3(data))
            output = output.sum(0)
            return output
