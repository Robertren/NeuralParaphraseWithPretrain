import numpy
import os
import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
import argparse
from util import *
from model import *


def calculate_aligned_phrases(phrases_a, phrases_b, biases, self_attention = False):
    """
    :param dot_products: A list of dot products that came from first attend feed forward network
    :param phrases: The phrases of a single word, NOTE: A sentence has been decomposed to a list of phrases
    :param biases: The bias for computing learned distance-sensitive bias term
    :return: sub phrase in b (or a) that is softly aligned to a_i or (b_i)
    """
    # Calculte attend score eij here
    dot_products = phrases_a.matmul(phrases_b.transpose(2, 1))
    # batch * paraphrase_length * paraphrase_length 64 * 20 * 20
    exp = torch.exp(dot_products)
    # 64 * 20 * 20
    if not self_attention:
        # 64 * 20
        # sum_beta = torch.sum(exp, 2)
        # sum_alpha = torch.sum(exp, 1)
        # div1 = exp / sum_beta.view(-1, 1)
        # div2 = exp / sum_alpha.view(1, -1)
        div1 = exp / exp.sum(2, keepdim=True)
        div2 = exp / exp.sum(1, keepdim=True)
        alpha, beta = torch.matmul(div2.transpose(2, 1), phrases_a), torch.matmul(div1, phrases_b)
    # else:
    return alpha, beta


def concat_representation(phrases, aligned_phrase):
    """
    :param phrases: The original phrase
    :param aligned_phrase: The aligned_phrase that attach to the origin
    :return: A concatenation representation
    """
    return torch.cat((phrases, aligned_phrase), 2)


def get_sentence_embedding(char_embedding, index):
    # input: [B, seq_len, 8]
    batch_size, seq_len, phrase_len = index.size()
    index = index.view(batch_size * seq_len, phrase_len)
    # [B*seq_len, 8]
    data = char_embedding(index)
    # [B*seq_len, 8, d]
    data = data.sum(1)
    # [B*seq_len, d]
    data = data.view(batch_size, seq_len, -1)
    return data


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


def train_model(model_list, num_epochs, optimizer, train_loader, test_loader, cuda, batch_size, length, model_dir, char_embedding):
    # unzip the models
    attend_model = model_list[0]
    self_attend_model = model_list[1]
    self_aligned_model = model_list[2]
    compare_model = model_list[3]
    aggregate_model = model_list[4]
    # Start training
    for epoch in range(num_epochs):
        # TODO: Save model torch.save(model.state_dict(), model_dir + "epoch{0}.pth".format(str(epoch)))
        for iter, (sentence_data_index_a, sentence_data_index_b, label) in enumerate(train_loader):
            sentence_data_index_a, sentence_data_index_b, label_batch = \
                Variable(sentence_data_index_a), Variable(sentence_data_index_b), Variable(label)
            if cuda:
                sentence_data_index_a, sentence_data_index_b, label_batch = \
                    sentence_data_index_a.cuda(), sentence_data_index_b.cuda(), label_batch.cuda()
            optimizer.zero_grad()

            # TODO: Just put the embedding here and transfer the index to representation

            sentence_data_a = get_sentence_embedding(sentence_data_index_a)
            sentence_data_b = get_sentence_embedding(sentence_data_index_b)
            # TODO: this is basically wrong
            # two sentence attend here
            attend_phrase_a = attend_model(sentence_data_a)
            attend_phrase_b = attend_model(sentence_data_b)


            # batch_size, phrase_size, out_size = batch_phrase_a.size()
            # TODO: Think about the
            alpha, beta = calculate_aligned_phrases(attend_phrase_a, attend_phrase_b, None, self_attention=False)

            # Self attend here
            # TODO: There is something wrong with the embedding here, think about it here
            self_attend_phrase_a = self_attend_model(sentence_data_a)
            self_attend_phrase_b = self_aligned_model(sentence_data_b)
            self_aligned_phrase_a = calculate_aligned_phrases(self_attend_phrase_a, self_attend_phrase_a,
                                                              bias, self_attention=True)
            self_aligned_phrase_b = calculate_aligned_phrases(self_attend_phrase_b, self_attend_phrase_b,
                                                              bias, self_attention=True)

            # Modify the input
            sentence_data_a = concat_representation(sentence_data_a, self_aligned_phrase_a)
            sentence_data_b = concat_representation(sentence_data_b, self_aligned_phrase_b)

            # Prepare for the compare
            concat_sentence_data_a = concat_representation(sentence_data_a, beta)
            concat_sentence_data_b = concat_representation(sentence_data_b, alpha)

            # Compare two sentence
            # TODO: rememeber to do summation here
            compared_sentence_data_a = compare_model(concat_sentence_data_a)
            compared_sentence_data_b = compare_model(concat_sentence_data_b)

            # Summation of two sentences
            summation_sentence_data_a = compared_sentence_data_a.sum(1)
            summation_sentence_data_b = compared_sentence_data_b.sum(1)

            # Prepare for the prediction
            sentence_representation = torch.cat((summation_sentence_data_a, summation_sentence_data_b), 1)

            # Got predict results
            outputs = aggregate_model(sentence_representation)

            loss = nn.functional.nll_loss(outputs, label_batch)
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
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hdf5_dir', type=str, default="../data/hdf5")
    parser.add_argument('--train_data_dir', type=str, default="../data/Quora_question_pair_partition/train.tsv")
    parser.add_argument('--test_data_dir', type=str, default="../data/Quora_question_pair_partition/test.tsv")
    parser.add_argument('--dev_data_dir', type=str, default="../data/Quora_question_pair_partition/dev.tsv")
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--label_encoder_dir', type=str, default="../data/french_test/label_encoder.bin")
    parser.add_argument('--model_dir', type=str, default="../model/french_test/fasttext")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_dir):
        # Each time change vocab size, should be careful here. maybe need to delete the hdf5 directory
        os.mkdir(args.hdf5_dir)
        train_data_set = construct_data_set(args.train_data_dir)
        test_data_set = construct_data_set(args.test_data_dir)
        dev_data_set = construct_data_set(args.dev_data_dir)

        processed_train_data, gram_indexer = process_text_dataset(train_data_set, args.window_size,
                                                                  args.ngram_n, args.vocab_size)
        processed_dev, _ = process_text_dataset(dev_data_set, args.window_size, args.ngram_n,
                                                ngram_indexer=gram_indexer)
        processed_test, _ = process_text_dataset(test_data_set, args.window_size, args.ngram_n,
                                                 ngram_indexer=gram_indexer)
        save_to_hdf5(processed_train_data, os.path.join(args.hdf5_dir, "train.hdf5"))
        save_to_hdf5(processed_dev, os.path.join(args.hdf5_dir, "dev.hdf5"))
        save_to_hdf5(processed_test, os.path.join(args.hdf5_dir, "test.hdf5"))
    # If there are hdf5 files saved, use them
    train_loader = construct_data_loader(os.path.join(args.hdf5_dir, "train.hdf5"), args.batch_size, shuffle=True)
    dev_loader = construct_data_loader(os.path.join(args.hdf5_dir, "dev.hdf5"), args.batch_size, shuffle=False)
    test_loader = construct_data_loader(os.path.join(args.hdf5_dir, "test.hdf5"), args.batch_size, shuffle=False)
    # Define models
    attend_model = AttendForwardNet(args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    self_attend_model = SelfAttentionForwardNet(args.embedding_size, args.hidden_size,
                                                args.output_size, args.dropout)
    self_aligned_model = SelfAttentionForwardNetAligned(args.embedding_size, args.hidden_size,
                                                        args.output_size, args.dropout)
    compare_model = CompareForwardNet(args.embedding_size, args.hidden_size,
                                      args.output_size, args.dropout)
    aggregate_model = AggregateForwardNet(args.embedding_size, args.hidden_size, args.hidden_size,
                                          args.output_size, args.dropout)
    if args.pretrained_embedding:
        char_embedding = torch.load(args.pretrained_embedding)
    else:
        char_embedding = nn.Embedding(args.vocab_size+1, embedding_dim= args.embedding_size, padding_idx=0)

    # Put model into the model lists
    model_list = [attend_model, self_attend_model, self_aligned_model, compare_model, aggregate_model]
    # Define optimizer
    optimizer = torch.optim.Adam(list(attend_model.parameters()) + list(self_attend_model.parameters()) +
                                 list(self_aligned_model.parameters()) + list(compare_model.parameters()) +
                                 list(aggregate_model.parameters()), lr=args.learning_rate)
    # start training

    train_model(model_list, args.num_epochs, optimizer, train_loader, dev_loader, False,
    args.batch_size, len(processed_train_data), args.model_dir)
