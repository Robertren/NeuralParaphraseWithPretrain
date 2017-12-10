import model
import argparse
import torch
import os
import util
import pickle

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
    parser.add_argument('--test_data_dir', type=str, default="../data/Quora_question_pair_partition/error.tsv")
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
    char_embedding = torch.load(args.pretrained_embedding)
    attend_model.load_state_dict(torch.load(os.path.join(args.model_dir, "attend_model_epoch49.pth")))
    self_attend_model.load_state_dict(torch.load(os.path.join(args.model_dir, "self_attend_model_epoch49.pth")))
    self_aligned_model.load_state_dict(torch.load(os.path.join(args.model_dir, "self_aligned_model_epoch49.pth")))
    compare_model.load_state_dict(torch.load(os.path.join(args.model_dir, "compare_model_epoch49.pth")))
    aggregate_model.load_state_dict(torch.load(os.path.join(args.model_dir, "aggregate_model_epoch49.pth")))
    model_list = [attend_model, self_attend_model, self_aligned_model, compare_model, aggregate_model]


