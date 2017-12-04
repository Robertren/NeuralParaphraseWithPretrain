from main import *
import argparse
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


def pretrain_model(model_list, num_epochs, optimizer, pretrain_loader, dev_loader,
                   cuda, batch_size, length, pre_emb_path, pre_char_embedding):
    for epoch in range(num_epochs):
        # TODO: Save model torch.save(model.state_dict(), model_dir + "epoch{0}.pth".format(str(epoch)))
        torch.save(pre_char_embedding, open(pre_emb_path + "_epoch{}.bin".format(str(epoch)), "wb"))
        for iter, (sentence_data_index_a, sentence_data_index_b, label) in tqdm(enumerate(pretrain_loader)):
            sentence_data_index_a, sentence_data_index_b, label_batch = \
                Variable(sentence_data_index_a), Variable(sentence_data_index_b), Variable(label)
            if cuda:
                sentence_data_index_a, sentence_data_index_b, label_batch = \
                    sentence_data_index_a.cuda(), sentence_data_index_b.cuda(), label_batch.cuda()
            optimizer.zero_grad()
            outputs = model_inference(model_list, sentence_data_index_a, sentence_data_index_b, pre_char_embedding, cuda)
            loss = nn.functional.binary_cross_entropy(outputs, label_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm(list(attend_model.parameters()) + list(self_attend_model.parameters()) +
                                          list(self_aligned_model.parameters()) + list(compare_model.parameters()) +
                                          list(aggregate_model.parameters()) + list(pre_char_embedding.parameters()),
                                          0.25)
            optimizer.step()
            if (iter + 1) % (batch_size * 64) == 0:
                print("Start testing")
        train_acc = test_model(pretrain_loader, model_list, 1, cuda, pre_char_embedding)
        print(train_acc)
        print("Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train Acc: {5}".format(
              epoch + 1, num_epochs, iter + 1, length // batch_size, loss.data[0], train_acc))
        dev_acc = test_model(dev_loader, model_list, 1, cuda, pre_char_embedding)
        print("Here is the validation set accuracy {}".format(str(dev_acc)))

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
    parser.add_argument('--vocab_path', type=str, default="../data/vocab/vocab_20000.pkl")
    parser.add_argument('--hdf5_dir', type=str, default="../data/hdf5")
    parser.add_argument('--pretained_data_dir', type=str, default="../data/Quora_question_pair_partition/paralex.tsv")
    parser.add_argument('--dev_data_dir', type=str, default="../data/Quora_question_pair_partition/dev.tsv")
    parser.add_argument('--pre_emb_path', type=str, default="../model/pre_trained_embedding")
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default="../model")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--pretrained', action='store_true', help='use CUDA')
    args = parser.parse_args()
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

    # Initialize pretrained embedding
    pre_char_embedding = nn.Embedding(args.vocab_size + 1, embedding_dim=args.embedding_size, padding_idx=0)
    pre_char_embedding.weight.data.uniform_(-1.0, 1.0)

    # set up cuda
    print(args.cuda)
    if args.cuda:
        print("Using GPU")
        attend_model.cuda()
        self_attend_model.cuda()
        self_aligned_model.cuda()
        compare_model.cuda()
        aggregate_model.cuda()
        pre_char_embedding.cuda()

    # Put model into the model lists
    model_list = [attend_model, self_attend_model, self_aligned_model, compare_model, aggregate_model]
    # Define optimizer
    # TODO; Think about here of number of models to optimize
    optimizer = torch.optim.Adam(list(attend_model.parameters()) + list(self_attend_model.parameters()) +
                                 list(self_aligned_model.parameters()) + list(compare_model.parameters()) +
                                 list(aggregate_model.parameters()) + list(pre_char_embedding.parameters()),
                                 lr=args.learning_rate)
    if not os.path.exists(os.path.join(args.hdf5_dir, "pretrained.hdf5")):
        dev_data_set = construct_data_set(args.dev_data_dir)
        pretrain_data_set = construct_data_set(args.pretained_data_dir)
        gram_indexer = pickle.load(open(args.vocab_path, "rb"))
        processed_pretrain, _ = process_text_dataset(pretrain_data_set, args.window_size,
                                                     args.ngram_n, ngram_indexer=gram_indexer)
        processed_dev, _ = process_text_dataset(dev_data_set, args.window_size, args.ngram_n,
                                                ngram_indexer=gram_indexer)
        save_to_hdf5(processed_pretrain, os.path.join(args.hdf5_dir, "pretrained.hdf5"))
    pretrain_loader = construct_data_loader(os.path.join(args.hdf5_dir, "pretrained.hdf5"),
                                            args.batch_size, shuffle=False)
    dev_loader = construct_data_loader(os.path.join(args.hdf5_dir, "dev.hdf5"), args.batch_size, shuffle=False)
    pretrain_model(model_list, args.num_epochs, optimizer, pretrain_loader,  dev_loader, args.cuda,
                   args.batch_size, len(pretrain_loader.dataset), args.pre_emb_path, pre_char_embedding)
