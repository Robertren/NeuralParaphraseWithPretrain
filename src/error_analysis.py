import model
import argparse

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
    parser.add_argument('--test_data_dir', type=str, default="../data/Quora_question_pair_partition/test.tsv")
    parser.add_argument('--dev_data_dir', type=str, default="../data/Quora_question_pair_partition/dev.tsv")
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--label_encoder_dir', type=str, default="../data/french_test/label_encoder.bin")
    parser.add_argument('--model_dir', type=str, default="../model")
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