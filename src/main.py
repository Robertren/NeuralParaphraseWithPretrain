import numpy
import torch
import pickle
from torch.autograd import Variable
import argparse
from .util import *
from .model import *
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



def test_model(loader, model, k, cuda):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
        """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = Variable(data), Variable(lengths), Variable(labels)
        if cuda:
            data_batch, length_batch, label_batch = data_batch.cuda(), length_batch.cuda(), label_batch.cuda()
        outputs = model(data_batch, length_batch)
        predicted = torch.topk(outputs.data, k, dim=1)[1]
        predicted = predicted.cpu().numpy()
        label_batch = label_batch.cpu().data.numpy()
        total += len(label_batch)
        correct += sum([label_batch[i] in predicted[i] for i in range(len(label_batch))])
        model.train()
    return 100 * correct / total

def train_model(num_epochs, optimizer, train_loader, test_loader, cuda, batch_size, length, model_dir):
    for epoch in range(num_epochs):
        # TODO: Save model torch.save(model.state_dict(), model_dir + "epoch{0}.pth".format(str(epoch)))
        for sentence_data_a, sentence_data_b, labels in enumerate(train_loader):
            sentence_data_a, sentence_data_b, label_batch = Variable(sentence_data_a), Variable(sentence_data_b), Variable(labels)
            if cuda:
                data_batch, length_batch, label_batch = data_batch.cuda(), length_batch.cuda(), label_batch.cuda()
            optimizer.zero_grad()
            # nn.squential look
            outputs = model_list(sentence_data_a, sentence_data_b)
            loss = nn.functional.binary_cross_entropy(outputs, label_batch)
            loss.backward()
            optimizer.step()
            # check here

            # if (i + 1) % (batch_size * 4) == 0:
            #     train_acc = test_model(train_loader, model, 1, cuda)
            #     print("Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train Acc: {5}".format(
            #         epoch + 1, num_epochs, i + 1, length // batch_size, loss.data[0], train_acc))
            #     print('The gradient is {}'.format(str(calculate_gradient(model))))
            #     test_acc = test_model(test_loader, model, 1, cuda)
            #     print(test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ngram_n', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--output_size', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--train_data_dir', type=str, default="../data/Quora_question_pair_partition/train.tsv")
    parser.add_argument('--test_data_dir', type=str, default="../data/Quora_question_pair_partition/test.tsv")
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--label_encoder_dir', type=str, default="../data/french_test/label_encoder.bin")
    parser.add_argument('--model_dir', type=str, default="../model/french_test/fasttext")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    args = parser.parse_args()
    train_data_set = construct_data_set(args.train_data_dir)
    test_data_set = construct_data_set(args.test_data_dir)
    processed_train_data, gram_indexer = process_text_dataset(train_data_set, args.window_size, args.ngram_n)
    processed_test, _ = process_text_dataset(test_data_set, args.window_size, args.ngram_n, ngram_indexer=gram_indexer)
    train_loader = construct_data_loader(processed_train_data, args.batch_size, shuffle=True)
    test_loader = construct_data_loader(processed_test, args.batch_size, shuffle=False)
    attend_model = AttendForwardNet(len(gram_indexer), args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    self_attend_model = SelfAttentionForwardNet(args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    self_aligned_model = SelfAttentionForwardNetAligned(args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    compare_model = CompareForwardNet(args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    aggregate_model = AggregateForwardNet(args.embedding_size, args.hidden_size, args.hidden_size, args.output_size, args.dropout)

    model_list = [attend_model, self_attend_model, self_aligned_model, compare_model, aggregate_model]
    optimizer = torch.optim.Adam([model.parameters() for model in model_list], lr=args.learning_rate)
    train_model(args.num_epochs, optimizer, train_loader, test_loader, args.cuda,
    args.batch_size, len(processed_train_data), args.model_dir)
