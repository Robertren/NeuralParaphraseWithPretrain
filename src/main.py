import numpy
import torch
import pickle
import argparse
from util import *
# from model import *
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


def train_model(num_epochs, optimizer, train_loader, test_loader, cuda, batch_size, data_length, model_dir):
    step = 0
    epoch = 0
    total_batches = int(len(training_set) / batch_size)
    while epoch <= num_epochs:
        model.train()
        vectors, labels = get_batch(next(training_iter))
        vectors = Variable(torch.stack(vectors).squeeze())  # batch_size, seq_len
        labels = Variable(torch.stack(labels).squeeze())

        model.zero_grad()

        if lstm:
            hidden, c_t = model.init_hidden()
            output, hidden = model(vectors, hidden, c_t)
        else:
            hidden = model.init_hidden()
            output, hidden = model(vectors, hidden)

        lossy = loss_(output, labels)
        lossy.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optim.step()

        if step % total_batches == 0:
            if epoch % 5 == 0:
                print("Epoch %i; Step %i; Loss %f; Train acc: %f; Dev acc %f"
                      % (epoch, step, lossy.data[0], \
                         evaluate(model, train_eval_iter, lstm), \
                         evaluate(model, dev_iter, lstm)))
            epoch += 1
        step += 1



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
    args.batch_size
    #train_data_set = construct_data_set(args.train_data_dir)
    test_data_set = construct_data_set(args.test_data_dir)
    processed_train_data, gram_indexer = process_text_dataset(test_data_set, args.window_size, args.ngram_n)
    #processed_test, _ = process_text_dataset(test_data_set, args.window_size, args.ngram_n, ngram_indexer=gram_indexer)
    train_loader = construct_data_loader(processed_train_data, args.batch_size, shuffle=True)
    #test_loader = construct_data_loader(processed_test, args.batch_size, shuffle=False)
    attend_model = AttendForwardNet(len(gram_indexer), args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    self_attend_model = SelfAttentionForwardNet(len(gram_indexer), args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    self_aligned_model = SelfAttentionForwardNetAligned(len(gram_indexer), args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    compare_model = CompareForwardNet(len(gram_indexer), args.embedding_size, args.hidden_size, args.output_size, args.dropout)
    aggregate_model = AggregateForwardNet(len(gram_indexer), args.embedding_size, args.hidden_size, args.hidden_size, args.output_size, args.dropout)

    model_list = [attend_model, self_attend_model, self_aligned_model, compare_model, aggregate_model]
    optimizer = torch.optim.Adam([model.parameters() for model in model_list], lr=args.learning_rate)
    train_model(args.num_epochs, optimizer, train_loader, test_loader, args.cuda,
                args.batch_size, len(processed_train_data), args.model_dir)
