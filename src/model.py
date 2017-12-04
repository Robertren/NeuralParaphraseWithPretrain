import torch
import torch.nn as nn


class AttendForwardNet(nn.Module):
    """
    First Attend Feed Forward Network to calculate e_ij
    """
    def __init__(self, embedding_size, hidden_size, output_size, p):
        super(AttendForwardNet, self).__init__()
        self.attend_linear1 = nn.Linear(embedding_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.attend_linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, data):
        data = self.dropout(self.attend_linear1(data))
        data = self.relu(data)
        data = self.batch_norm(data)
        data = self.dropout(self.attend_linear2(data))
        # [Batch_size, MAX_phrase_length ,output_size]
        return data


class SelfAttentionForwardNet(nn.Module):
    """
    The model which help calculate self attention to encode relationships between words with each sentence
    """
    def __init__(self, embedding_size, hidden_size, output_size, p):
        super(SelfAttentionForwardNet, self).__init__()
        self.attention_linear1 = nn.Linear(embedding_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.attention_linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, data):
        data = self.dropout(self.attention_linear1(data))
        data = self.relu(data)
        data = self.batch_norm(data)
        data = self.dropout(self.attention_linear2(data))
        return data


class SelfAttentionForwardNetAligned(nn.Module):
    """
    TODO: Discuss if we need to delete this model
    This model can be the same with the above one
    The model which help calculate self attention to encode relationships between words with each sentence
    """

    def __init__(self, embedding_size, hidden_size, output_size, p):
        super(SelfAttentionForwardNetAligned, self).__init__()
        self.attention_aligned_linear1 = nn.Linear(embedding_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.attention_aligned_linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, data):
        data = self.dropout(self.attention_aligned_linear1(data))
        data = self.relu(data)
        data = self.batch_norm(data)
        data = self.dropout(self.attention_aligned_linear2(data))
        return data


class CompareForwardNet(nn.Module):
    """
    This is the feed forward network to compare aligned phrases
    """

    def __init__(self, compare_emb_size, hidden_size, output_size, p):
        super(CompareForwardNet, self).__init__()
        self.compare_linear1 = nn.Linear(compare_emb_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.compare_linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, data):
        data = self.dropout(self.compare_linear1(data))
        data = self.relu(data)
        data = self.batch_norm(data)
        data = self.dropout(self.compare_linear2(data))
        return data


class AggregateForwardNet(nn.Module):
        """
        This is the feed forward network to aggregate all the outputs from the compare network and
        give a prediction of the label.
        """

        def __init__(self, agg_emb_size, hidden_size1, hidden_size2, label_size, p=0.1):
            super(AggregateForwardNet, self).__init__()
            self.agg_linear1 = nn.Linear(agg_emb_size, hidden_size1)
            self.relu = nn.ReLU()
            self.agg_linear2 = nn.Linear(hidden_size1, hidden_size2)
            self.agg_linear3 = nn.Linear(hidden_size2, label_size)
            self.dropout = nn.Dropout(p)

        def forward(self, data):
            data = self.dropout(self.agg_linear1(data))
            data = self.relu(data)
            data = self.dropout(self.agg_linear2(data))
            output = nn.functional.sigmoid(self.agg_linear3(data.float()))
            return output.view(-1)
