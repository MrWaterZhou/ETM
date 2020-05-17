import torch
from etm import ETM
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')

parser.add_argument('--load_from', type=str,
                    default='results/etm_koa_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0',
                    help='the name of the ckpt to eval from')
parser.add_argument('--emb_path', type=str, default='data/koa/embeddings', help='directory containing word embeddings')
parser.add_argument('--vocab_path', type=str, default='data/koa/vocab.pkl')
parser.add_argument('--corpus', type=str, default='')
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')


def get_batch(corpus, vocab_dict:dict, ind, vocab_size, device, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))

    tokens = []
    counts = []

    for words in corpus:
        count_dict = {}
        for w in words:
            if w in vocab_dict:
                idx = vocab_dict[w]
                if idx not in count_dict:
                    count_dict[idx] = 0
                count_dict[idx] += 1
        token = np.array(count_dict.keys())
        count = np.array([count_dict[i] for i in token])
        tokens.append(token)
        counts.append(count)


    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        L = count.shape[1]
        if len(doc) == 1:
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch

if __name__ == '__main__':
    args = parser.parse_args()
    ckpt = args.load_from
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    vocab_dict = {w:i for i,w in enumerate(vocab)}
    vocab_size = len(vocab_dict)

    corpus = open(args.corpus, 'r').readlines()
    corpus = [x.strip().split() for x in corpus]



    with torch.no_grad():
        ## show topics
        topic_represent = []
        beta = model.get_beta()
        topic_indices = list(np.random.choice(args.num_topics, 10))  # 10 random topics
        print('\n')
        for k in range(args.num_topics):  # topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words + 1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            print('Topic {}: {}'.format(k, topic_words))
            topic_represent.append(topic_words)

        ## get most used topics
        indices = torch.tensor(range(args.num_docs_train))
        indices = torch.split(indices, args.batch_size)
        thetaAvg = torch.zeros(1, args.num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = get_batch(corpus, vocab_dict, ind, vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
            weighed_theta = sums * theta
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
                for row, th in zip(corpus[ind], theta):
                    topic = th.argsort().cpu().numpy()[::-1][0]
                    topic_re = topic_represent[int(topic)]
                    print("corpus:{}\n topic:{}\n pred:{}\n".format(row, topic_re, th[int(topic)]))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))
