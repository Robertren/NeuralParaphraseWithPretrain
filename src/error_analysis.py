import model
import argparse
import torch
import os
import util
import main
import pickle
from tqdm import tqdm
from torch.autograd import Variable


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
    parser.add_argument('--pretrained_dir', type=str, default="../model/final/char_embedding_epoch49.bin")
    parser.add_argument('--test_data_dir', type=str, default="../data/Quora_question_pair_partition/subset.tsv")
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--label_encoder_dir', type=str, default="../data/french_test/label_encoder.bin")
    parser.add_argument('--model_dir', type=str, default="../model/final")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--pretrained', action='store_true', help='use CUDA')
    # Define models
    args = parser.parse_args()
    attend_model = model.AttendForwardNet(args.embedding_size,
                                    args.hidden_size,
                                    args.output_size,
                                    args.dropout)
    self_attend_model = model.SelfAttentionForwardNet(args.embedding_size,
                                                args.hidden_size,
                                                args.output_size,
                                                args.dropout)
    self_aligned_model = model.SelfAttentionForwardNetAligned(args.embedding_size,
                                                              args.hidden_size,
                                                        args.output_size,
                                                        args.dropout)
    compare_model = model.CompareForwardNet(args.embedding_size * 3,
                                            args.hidden_size,
                                            args.output_size,
                                            args.dropout)
    aggregate_model = model.AggregateForwardNet(args.output_size * 2,
                                                args.hidden_size,
                                                args.output_size,
                                                args.label_size,
                                                args.dropout)
    # build data loader for error analysis
    error_analysis_data_set = util.construct_data_set(args.test_data_dir)
    gram_indexer = pickle.load(open(args.vocab_path + "_" + str(args.vocab_size) + ".pkl", "rb"))
    processed_error_data, _ = util.process_text_dataset(error_analysis_data_set, args.window_size,
                                                        args.ngram_n, args.vocab_size,
                                                        ngram_indexer=gram_indexer)
    error_data_loader = util.save_to_hdf5(processed_error_data, os.path.join(args.hdf5_dir, "error.hdf5"))
    error_loader = util.construct_data_loader(os.path.join(args.hdf5_dir, "error.hdf5"), 1, shuffle=True)

    # Load trained model
    char_embedding = torch.load(args.pretrained_dir, map_location=lambda storage, loc: storage)
    attend_model.load_state_dict(torch.load(os.path.join(args.model_dir, "attend_model_epoch49.pth"),
                                            map_location=lambda storage, loc: storage))
    self_attend_model.load_state_dict(torch.load(os.path.join(args.model_dir, "self_attend_model_epoch49.pth"),
                                                 map_location=lambda storage, loc: storage))
    self_aligned_model.load_state_dict(torch.load(os.path.join(args.model_dir, "self_aligned_model_epoch49.pth"),
                                                  map_location=lambda storage, loc: storage))
    compare_model.load_state_dict(torch.load(os.path.join(args.model_dir, "compare_model_epoch49.pth"),
                                             map_location=lambda storage, loc: storage))
    aggregate_model.load_state_dict(torch.load(os.path.join(args.model_dir, "aggregate_model_epoch49.pth"),
                                               map_location=lambda storage, loc: storage))
    model_list = [attend_model, self_attend_model, self_aligned_model, compare_model, aggregate_model]


    # Get the output for dataloader
    true_labels = []
    predictions = []
    attention_matrix_list = []
    for iter, (sentence_data_index_a, sentence_data_index_b, label) in tqdm(enumerate(error_loader)):
        sentence_data_index_a, sentence_data_index_b, label_batch = \
            Variable(sentence_data_index_a), Variable(sentence_data_index_b), Variable(label)
        outputs, attention_matrix = main.model_inference(model_list, sentence_data_index_a,
                                                         sentence_data_index_b, char_embedding, False)
        attention_matrix_list.append(attention_matrix)
        prediction = 1 if outputs.data[0] > 0.3 else 0
        predictions.append(prediction)
        true_labels.append(label_batch.data[0])
    pickle.dump(attention_matrix_list, open("../results/attention_matrix.pkl", "wb"))
    print(predictions)
    print(true_labels)
