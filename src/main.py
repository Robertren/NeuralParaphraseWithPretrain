import numpy
import os
import torch
import torch.nn as nn
import pickle
from torch.autograd import Variable
import argparse
from util import *
from model import *
from tqdm import tqdm


def model_inference(model_list, sentence_data_index_a, sentence_data_index_b, char_embedding, cuda):
    # unzip the models
    attend_model = model_list[0]
    self_attend_model = model_list[1]
    self_aligned_model = model_list[2]
    compare_model = model_list[3]
    aggregate_model = model_list[4]

    sentence_data_a = get_sentence_embedding(char_embedding, index=sentence_data_index_a, cuda=cuda)
    sentence_data_b = get_sentence_embedding(char_embedding, index=sentence_data_index_b, cuda=cuda)

    # two sentence attend here
    attend_phrase_a = attend_model(sentence_data_a)
    attend_phrase_b = attend_model(sentence_data_b)

    # batch_size, phrase_size, out_size = batch_phrase_a.size()
    alpha, beta = calculate_aligned_phrases(attend_phrase_a, attend_phrase_b, sentence_data_a,
                                            sentence_data_b, None, self_attention=False)

    # Self attend here
    self_attend_phrase_a = self_attend_model(sentence_data_a)
    self_attend_phrase_b = self_aligned_model(sentence_data_b)
    # TODO: Bias to look at here
    self_aligned_phrase_a, _ = calculate_aligned_phrases(self_attend_phrase_a, self_attend_phrase_a,
                                                         sentence_data_a, sentence_data_a,
                                                         None, self_attention=False)
    self_aligned_phrase_b, _ = calculate_aligned_phrases(self_attend_phrase_b, self_attend_phrase_b,
                                                         sentence_data_b, sentence_data_b,
                                                         None, self_attention=False)

    # Modify the input
    updated_sentence_data_a = concat_representation(sentence_data_a, self_aligned_phrase_a)
    updated_sentence_data_b = concat_representation(sentence_data_b, self_aligned_phrase_b)

    # Prepare for the compare
    concat_sentence_data_a = concat_representation(updated_sentence_data_a, beta)
    concat_sentence_data_b = concat_representation(updated_sentence_data_b, alpha)

    # Compare two sentence
    # TODO: rememeber to do summation here
    compared_sentence_data_a = compare_model(concat_sentence_data_a)
    compared_sentence_data_b = compare_model(concat_sentence_data_b)

    # Summation of two sentences
    summation_sentence_data_a = compared_sentence_data_a.sum(1)
    summation_sentence_data_b = compared_sentence_data_b.sum(1)
    sentence_representation = torch.cat((summation_sentence_data_a, summation_sentence_data_b), 1)
    outputs = aggregate_model(sentence_representation)

    return outputs


def calculate_aligned_phrases(phrases_a, phrases_b, origin_phrases_a, origin_phrases_b, biases, self_attention = False):
    """
    :param dot_products: A list of dot products that came from first attend feed forward network
    :param phrases: The phrases of a single word, NOTE: A sentence has been decomposed to a list of phrases
    :param biases: The bias for computing learned distance-sensitive bias term
    :return: sub phrase in b (or a) that is softly aligned to a_i or (b_i)
    """
    # Calculate attend score eij here
    dot_products = phrases_a.matmul(phrases_b.transpose(2, 1))
    # batch * paraphrase_length * paraphrase_length 64 * 20 * 20
    # Normalize the dot product matrix to get rid of model gradient exploding
    dot_products = nn.functional.normalize(dot_products, p=1, dim=1)
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
        alpha, beta = torch.matmul(div2.transpose(2, 1), origin_phrases_a), torch.matmul(div1, origin_phrases_b)
    # else:
    return alpha, beta


def concat_representation(phrases, aligned_phrase):
    """
    :param phrases: The original phrase
    :param aligned_phrase: The aligned_phrase that attach to the origin
    :return: A concatenation representation
    """
    return torch.cat((phrases, aligned_phrase), 2)


def get_sentence_embedding(char_embedding, index, cuda):
    """
    A function which get sentence char embedding for a batch of index
    :param char_embedding: A pretrained or initialized character level embedding
    :param index:
    :return:
    """
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


def calculate_gradient(model_list):
    """
    Helper function to calculate all the gradient in the model list
    :param model_list:
    :return:
    """
    gradient = 0
    for model in model_list:
        for name, w in model.named_parameters():
            if w.grad is not None:
                w_grad = torch.norm(w.grad.data, 2) ** 2
                gradient += w_grad
            else:
                print(name, w.grad)
    return gradient


def change_model_mode(model_list, mode):
    if mode == "eval":
        for model in model_list:
            model.eval()
    else:
        for model in model_list:
            model.train()


def test_model(loader, model_list, k, cuda, char_embedding):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
        """
    correct = 0
    total = 0
    change_model_mode(model_list, "eval")
    for iter, (sentence_data_index_a, sentence_data_index_b, label) in enumerate(loader):
        sentence_data_index_a, sentence_data_index_b, label_batch = \
            Variable(sentence_data_index_a), Variable(sentence_data_index_b), Variable(label)
        if cuda:
            sentence_data_index_a, sentence_data_index_b, label_batch = \
                sentence_data_index_a.cuda(), sentence_data_index_b.cuda(), label_batch.cuda()
        outputs = model_inference(model_list, sentence_data_index_a, sentence_data_index_b, char_embedding, cuda)
        predicted = (outputs.data > 0.3).long().view(-1)
        total += label_batch.size(0)
        correct += (predicted == label_batch.data).sum()
    change_model_mode(model_list, "train")
    print(correct, total)
    return 100 * correct / total


def save_all_model_parameters(model_list, model_dir, char_embedding, epoch):
    model_names = {0: "attend_model", 1: "self_attend_model", 2: "self_aligned_model",
                   3: "compare_model", 4: "aggregate_model"}
    for index in range(len(model_list)):
        torch.save(model_list[index].state_dict(), os.path.join(model_dir, model_names[index]) +
                   "_epoch{0}.pth".format(str(epoch)))
    torch.save(char_embedding, os.path.join(model_dir, "char_embedding_epoch{}.bin".format(str(epoch))))


def train_model(model_list, num_epochs, optimizer, train_loader, test_loader, cuda, batch_size, length, model_dir, char_embedding):
    # Start training
    for epoch in range(num_epochs):
        # TODO: Save model torch.save(model.state_dict(), model_dir + "epoch{0}.pth".format(str(epoch)))
        save_all_model_parameters(model_list, model_dir, char_embedding, epoch)
        for iter, (sentence_data_index_a, sentence_data_index_b, label) in tqdm(enumerate(train_loader)):
            sentence_data_index_a, sentence_data_index_b, label_batch = \
                Variable(sentence_data_index_a), Variable(sentence_data_index_b), Variable(label)
            if cuda:
                sentence_data_index_a, sentence_data_index_b, label_batch = \
                    sentence_data_index_a.cuda(), sentence_data_index_b.cuda(), label_batch.cuda()
            optimizer.zero_grad()
            outputs = model_inference(model_list, sentence_data_index_a, sentence_data_index_b, char_embedding, cuda)
            loss = nn.functional.binary_cross_entropy(outputs, label_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm(list(attend_model.parameters()) + list(self_attend_model.parameters()) +
                                          list(self_aligned_model.parameters()) + list(compare_model.parameters()) +
                                          list(aggregate_model.parameters()) + list(char_embedding.parameters()), 0.25)
            optimizer.step()
            if (iter + 1) % (batch_size*32) == 0:
                print("Start testing")
                train_acc = test_model(train_loader, model_list, 1, cuda, char_embedding)
                print(train_acc)
                print("Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train Acc: {5}".format(
                   epoch + 1, num_epochs, iter + 1, length // batch_size, loss.data[0], train_acc))
                print('The gradient is {}'.format(str(calculate_gradient(model_list+[char_embedding]))))
                test_acc = test_model(test_loader, model_list, 1, cuda, char_embedding)
                print("Here is the test acc")
                print(test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ngram_n', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--output_size', type=int, default=200)
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--label_size', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pretrained_embedding', type=bool, default=False)
    parser.add_argument('--vocab_path', type=str, default="../data/vocab/vocab")
    parser.add_argument('--hdf5_dir', type=str, default="../data/hdf5")
    parser.add_argument('--pretrained_dir', type=str, default="../model/pretrained")
    parser.add_argument('--train_data_dir', type=str, default="../data/Quora_question_pair_partition/train.tsv")
    parser.add_argument('--test_data_dir', type=str, default="../data/Quora_question_pair_partition/test.tsv")
    parser.add_argument('--dev_data_dir', type=str, default="../data/Quora_question_pair_partition/dev.tsv")
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--label_encoder_dir', type=str, default="../data/french_test/label_encoder.bin")
    parser.add_argument('--model_dir', type=str, default="../model")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--pretrained', action='store_true', help='use CUDA')
    args = parser.parse_args()
    torch.manual_seed(1)
    if not os.path.exists(args.hdf5_dir):
        # Each time change vocab size, should be careful here. maybe need to delete the hdf5 directory
        os.mkdir(args.hdf5_dir)
        train_data_set = construct_data_set(args.train_data_dir)
        test_data_set = construct_data_set(args.test_data_dir)
        dev_data_set = construct_data_set(args.dev_data_dir)
        if os.path.exists(args.vocab_path + "_" + str(args.vocab_size) + ".pkl"):
            gram_indexer = pickle.load(open(args.vocab_path + "_" + str(args.vocab_size) + ".pkl", "rb"))
            processed_train_data, _ = process_text_dataset(train_data_set, args.window_size,
                                                           args.ngram_n, args.vocab_size,
                                                           ngram_indexer=gram_indexer)
        else:
            processed_train_data, gram_indexer = process_text_dataset(train_data_set, args.window_size,
                                                                      args.ngram_n, args.vocab_size)
            pickle.dump(gram_indexer, open(args.vocab_path + "_" + str(args.vocab_size) + ".pkl", "wb"))
        processed_dev, _ = process_text_dataset(dev_data_set, args.window_size, args.ngram_n,
                                                ngram_indexer=gram_indexer)
        processed_test, _ = process_text_dataset(test_data_set, args.window_size, args.ngram_n,
                                                 ngram_indexer=gram_indexer)
        save_to_hdf5(processed_train_data, os.path.join(args.hdf5_dir, "train.hdf5"))
        save_to_hdf5(processed_dev, os.path.join(args.hdf5_dir, "dev.hdf5"))
        save_to_hdf5(processed_test, os.path.join(args.hdf5_dir, "test.hdf5"))

    # If there are hdf5 files saved, use them
    train_loader = construct_data_loader(os.path.join(args.hdf5_dir, "train.hdf5"), args.batch_size, shuffle=True)
    dev_loader = construct_data_loader(os.path.join(args.hdf5_dir, "dev.hdf5"),  args.batch_size, shuffle=False)
    test_loader = construct_data_loader(os.path.join(args.hdf5_dir, "test.hdf5"), args.batch_size, shuffle=False)

    # Define models
    attend_model = AttendForwardNet(args.embedding_size,
                                    args.hidden_size,
                                    args.output_size,
                                    args.dropout)
    self_attend_model = SelfAttentionForwardNet(args.embedding_size,
                                                args.hidden_size,
                                                args.output_size,
                                                args.dropout)
    self_aligned_model = SelfAttentionForwardNetAligned(args.embedding_size,
                                                        args.hidden_size,
                                                        args.output_size,
                                                        args.dropout)
    compare_model = CompareForwardNet(args.embedding_size * 3,
                                      args.hidden_size,
                                      args.output_size,
                                      args.dropout)
    aggregate_model = AggregateForwardNet(args.output_size * 2,
                                          args.hidden_size,
                                          args.output_size,
                                          args.label_size,
                                          args.dropout)
    # Get embeddings
    if args.pretrained_embedding:
        char_embedding = torch.load(args.pretrained_embedding)
        attend_model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "attend_model_epoch10.pth")))
        self_attend_model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "self_attend_model_epoch10.pth")))
        self_aligned_model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "self_aligned_model_epoch10.pth")))
        compare_model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "compare_model_epoch10.pth")))
        aggregate_model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "aggregate_model_epoch10.pth")))
    else:
        char_embedding = nn.Embedding(args.vocab_size+1, embedding_dim=args.embedding_size, padding_idx=0)
        char_embedding.weight.data.uniform_(-1.0, 1.0)

    print(args.cuda)
    if args.cuda:
        print("Using GPU")
        attend_model.cuda()
        self_attend_model.cuda()
        self_aligned_model.cuda()
        compare_model.cuda()
        aggregate_model.cuda()
        char_embedding.cuda()


    # Put model into the model lists
    model_list = [attend_model, self_attend_model, self_aligned_model, compare_model, aggregate_model]
    # Define optimizer
    # TODO; Think about here of number of models to optimize
    optimizer = torch.optim.Adam(list(attend_model.parameters()) + list(self_attend_model.parameters()) +
                                 list(self_aligned_model.parameters()) + list(compare_model.parameters()) +
                                 list(aggregate_model.parameters()) + list(char_embedding.parameters()),
                                 lr=args.learning_rate)
    # optimizer = torch.optim.Adam(list(compare_model.parameters()) + list(aggregate_model.parameters()) +
    #                              list(char_embedding.parameters()), lr=args.learning_rate)

    # start training
    # TODO: DEBUGGING script here. Using
    # train_model(model_list, args.num_epochs, optimizer, dev_loader, test_loader, args.cuda,
    #            args.batch_size, len(dev_loader.dataset), args.model_dir, char_embedding)

    train_model(model_list, args.num_epochs, optimizer, train_loader, dev_loader, args.cuda,
                args.batch_size, len(train_loader.dataset), args.model_dir, char_embedding)

