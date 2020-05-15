import sys
from scipy.io import savemat
import os
import pickle

def convert_to_bow(vocab: dict, segment: list):
    result = {}
    for word in segment:
        if word in vocab:
            if vocab[word] not in result:
                result[vocab[word]] = 0
            result[vocab[word]] += 1
    tokens = list(result.keys())
    counts = [result[token] for token in tokens]
    return tokens, counts


if __name__ == '__main__':
    vocab = open(sys.argv[1], 'r').readlines()
    vocab_pkl = list([line.strip().split(' ')[1] for line in vocab if len(line.strip().split(' '))==2])
    vocab = {w: i for i, w in enumerate(vocab_pkl)}
    segments = open(sys.argv[2], 'r').readlines()
    path_save = sys.argv[3]

    with open(os.path.join(path_save , 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab_pkl, f)


    tokens_agg = []
    counts_agg = []
    for line in segments:
        tokens, counts = convert_to_bow(vocab, line.strip().split())
        tokens_agg.append(tokens)
        counts_agg.append(counts_agg)
    train_end = int(0.8 * len(tokens_agg))
    valid_end = int(0.9 * len(tokens_agg))

    savemat(os.path.join(path_save, 'bow_tr_tokens'), {'tokens': tokens_agg[:train_end]}, do_compression=True)
    savemat(os.path.join(path_save, 'bow_tr_counts'), {'counts': counts_agg[:train_end]}, do_compression=True)

    savemat(os.path.join(path_save, 'bow_ts_tokens'), {'tokens': tokens_agg[train_end:valid_end]}, do_compression=True)
    savemat(os.path.join(path_save, 'bow_ts_counts'), {'counts': counts_agg[train_end:valid_end]}, do_compression=True)

    savemat(os.path.join(path_save, 'bow_ts_h1_tokens'), {'tokens': tokens_agg[train_end:valid_end]}, do_compression=True)
    savemat(os.path.join(path_save, 'bow_ts_h1_counts'), {'counts': counts_agg[train_end:valid_end]}, do_compression=True)

    savemat(os.path.join(path_save, 'bow_ts_h2_tokens'), {'tokens': tokens_agg[train_end:valid_end]}, do_compression=True)
    savemat(os.path.join(path_save, 'bow_ts_h2_counts'), {'counts': counts_agg[train_end:valid_end]}, do_compression=True)

    savemat(os.path.join(path_save, 'bow_va_tokens'), {'tokens': tokens_agg[valid_end:]}, do_compression=True)
    savemat(os.path.join(path_save, 'bow_va_counts'), {'counts': counts_agg[valid_end:]}, do_compression=True)
