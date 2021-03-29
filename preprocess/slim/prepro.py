import os
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize

def tokenize_data(data, word_count=False):
    '''
    Tokenize captions, questions and answers
    Also maintain word count if required
    '''
    res, word_counts = {}, {}

    print(data['split'])
    print('Tokenizing captions...')
    
    for i in data['data']['dialogs']:
        img_id = i['image_id']
        caption = word_tokenize(i['caption'])
        res[img_id] = {'caption': caption}

    print('Tokenizing questions...')
    ques_toks, ans_toks = [], []
    for i in data['data']['questions']:
        ques_toks.append(word_tokenize(i + '?'))
    print('Tokenizing answers...')
    for i in data['data']['answers']:
        ans_toks.append(word_tokenize(i))

    for i in data['data']['dialogs']:
        res[i['image_id']]['dialog'] = i['dialog']
        for j in range(10):
            question = ques_toks[i['dialog'][j]['question']]
            answer = ans_toks[i['dialog'][j]['answer']]
            if word_count == True:
                for word in question + answer:
                    word_counts[word] = word_counts.get(word, 0) + 1
    # res[img_id] = {caption: ...; dialog: []} 
    return res, ques_toks, ans_toks, word_counts

def encode_vocab(data_toks, ques_toks, ans_toks, word2ind):
    '''
    Converts string tokens to indices based on given dictionary
    '''
    max_ques_len, max_ans_len, max_cap_len = 0, 0, 0
    for k, v in data_toks.items():
        image_id = k
        caption = [word2ind.get(word, word2ind['<UNK>']) \
                for word in v['caption']]
        if max_cap_len < len(caption): max_cap_len = len(caption)
        data_toks[k]['caption_inds'] = caption
        data_toks[k]['caption_len'] = len(caption)

    ques_inds, ans_inds = [], []
    for i in ques_toks:
        question = [word2ind.get(word, word2ind['<UNK>']) \
                for word in i]
        ques_inds.append(question)

    for i in ans_toks:
        answer = [word2ind.get(word, word2ind['<UNK>']) \
                for word in i]
        ans_inds.append(answer)

    return data_toks, ques_inds, ans_inds

def create_data_mats(data_toks, ques_inds, ans_inds, params):
    num_threads = len(data_toks.keys()) # number of images
    num_rounds = 10
    max_cap_len = params.max_cap_len
    max_ques_len = params.max_ques_len
    max_ans_len = params.max_ans_len

    captions = np.zeros([num_threads, max_cap_len])
    questions = np.zeros([num_threads, num_rounds, max_ques_len])
    answers = np.zeros([num_threads, num_rounds, max_ans_len])

    answer_index = np.zeros([num_threads, num_rounds])

    caption_len = np.zeros(num_threads, dtype=np.int)
    question_len = np.zeros([num_threads, num_rounds], dtype=np.int)
    answer_len = np.zeros([num_threads, num_rounds], dtype=np.int)

    image_index = np.zeros(num_threads)

    options = np.zeros([num_threads, num_rounds, 100]) # each image has 10 rounds and each round has 100 options 

    image_list = []
    for i in range(num_threads):
        image_id = data_toks.keys()[i]
        image_list.append(image_id)
        image_index[i] = i
        caption_len[i] = len(data_toks[image_id]['caption_inds'][0:max_cap_len])
        captions[i][0:caption_len[i]] = data_toks[image_id]['caption_inds'][0:max_cap_len]
        for j in range(10):
            question_len[i][j] = len(ques_inds[data_toks[image_id]['dialog'][j]['question']][0:max_ques_len])
            questions[i][j][0:question_len[i][j]] = ques_inds[data_toks[image_id]['dialog'][j]['question']][0:max_ques_len]
            answer_len[i][j] = len(ans_inds[data_toks[image_id]['dialog'][j]['answer']][0:max_ans_len])
            answers[i][j][0:answer_len[i][j]] = ans_inds[data_toks[image_id]['dialog'][j]['answer']][0:max_ans_len]
            answer_index[i][j] = data_toks[image_id]['dialog'][j]['gt_index']
            options[i][j] = np.array(data_toks[image_id]['dialog'][j]['answer_options'])

    options_list = np.zeros([len(ans_inds), max_ans_len])
    options_len = np.zeros(len(ans_inds), dtype=np.int)
    for i in range(len(ans_inds)):
        options_len[i] = len(ans_inds[i][0:max_ans_len])
        options_list[i][0:options_len[i]] = ans_inds[i][0:max_ans_len]

    return captions, caption_len, questions, question_len, answers, answer_len, options, options_list, options_len, answer_index, image_index, image_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-download', default=0, type=int, help='Whether to download VisDial v0.9 data')

    # Input files
    parser.add_argument('-input_json_train', default='./visdial_0.9_train.json', help='Input `train` json file')
    parser.add_argument('-input_json_val', default='./visdial_0.9_val.json', help='Input `val` json file')

    # Output files
    parser.add_argument('-output_json', default='visdial_params_2.json', help='Output json file')
    parser.add_argument('-output_h5', default='visdial_data_2.h5', help='Output hdf5 file')

    # Options
    parser.add_argument('-max_ques_len', default=30, type=int, help='Max length of questions') # 20
    parser.add_argument('-max_ans_len', default=30, type=int, help='Max length of answers') # 20
    parser.add_argument('-max_cap_len', default=60, type=int, help='Max length of captions') # 40
    parser.add_argument('-word_count_threshold', default=5, type=int, help='Min threshold of word count to include in vocabulary')

    args = parser.parse_args()

    if args.download == 1:
        os.system('wget https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_train.zip')
        os.system('wget https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_val.zip')

        os.system('unzip visdial_0.9_train.zip')
        os.system('unzip visdial_0.9_val.zip')

    print('Reading json...')
    data_train = json.load(open(args.input_json_train, 'r'))
    data_val = json.load(open(args.input_json_val, 'r'))

    # Tokenizing
    data_train_toks, ques_train_toks, ans_train_toks, word_counts_train = tokenize_data(data_train, True)
    data_val_toks, ques_val_toks, ans_val_toks, _ = tokenize_data(data_val)

    print('Building vocabulary...')
    _PAD = b"<PAD>"
    _GO = b"<START>"
    _EOS = b"<END>"
    _UNK = b"<UNK>"
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

    vocab = [word for word in word_counts_train \
            if word_counts_train[word] >= args.word_count_threshold]
    vocab = _START_VOCAB + vocab
    print('Words: %d' % len(vocab))
    word2ind = {word:word_ind for word_ind, word in enumerate(vocab)}
    ind2word = {word_ind:word for word, word_ind in word2ind.items()}

    print('Encoding based on vocabulary...')
    data_train_toks, ques_train_inds, ans_train_inds = encode_vocab(data_train_toks, ques_train_toks, ans_train_toks, word2ind)
    data_val_toks, ques_val_inds, ans_val_inds = encode_vocab(data_val_toks, ques_val_toks, ans_val_toks, word2ind)

    print('Creating data matrices...')
    captions_train, captions_train_len, questions_train, questions_train_len, answers_train, answers_train_len, options_train, options_train_list, options_train_len, answers_train_index, images_train_index, images_train_list = create_data_mats(data_train_toks, ques_train_inds, ans_train_inds, args)
    captions_val, captions_val_len, questions_val, questions_val_len, answers_val, answers_val_len, options_val, options_val_list, options_val_len, answers_val_index, images_val_index, images_val_list = create_data_mats(data_val_toks, ques_val_inds, ans_val_inds, args)

    print('Saving hdf5...')
    f = h5py.File(args.output_h5, 'w')
    f.create_dataset('ques_train', dtype='uint32', data=questions_train)
    f.create_dataset('ques_length_train', dtype='uint32', data=questions_train_len)
    f.create_dataset('ans_train', dtype='uint32', data=answers_train)
    f.create_dataset('ans_length_train', dtype='uint32', data=answers_train_len)
    f.create_dataset('ans_index_train', dtype='uint32', data=answers_train_index)
    f.create_dataset('cap_train', dtype='uint32', data=captions_train)
    f.create_dataset('cap_length_train', dtype='uint32', data=captions_train_len)
    f.create_dataset('opt_train', dtype='uint32', data=options_train)
    f.create_dataset('opt_length_train', dtype='uint32', data=options_train_len)
    f.create_dataset('opt_list_train', dtype='uint32', data=options_train_list)
    f.create_dataset('img_pos_train', dtype='uint32', data=images_train_index)

    f.create_dataset('ques_val', dtype='uint32', data=questions_val)
    f.create_dataset('ques_length_val', dtype='uint32', data=questions_val_len)
    f.create_dataset('ans_val', dtype='uint32', data=answers_val)
    f.create_dataset('ans_length_val', dtype='uint32', data=answers_val_len)
    f.create_dataset('ans_index_val', dtype='uint32', data=answers_val_index)
    f.create_dataset('cap_val', dtype='uint32', data=captions_val)
    f.create_dataset('cap_length_val', dtype='uint32', data=captions_val_len)
    f.create_dataset('opt_val', dtype='uint32', data=options_val)
    f.create_dataset('opt_length_val', dtype='uint32', data=options_val_len)
    f.create_dataset('opt_list_val', dtype='uint32', data=options_val_list)
    f.create_dataset('img_pos_val', dtype='uint32', data=images_val_index)

    f.close()

    out = {}
    out['ind2word'] = ind2word
    out['word2ind'] = word2ind
    out['unique_img_train'] = images_train_list
    out['unique_img_val'] = images_val_list

    json.dump(out, open(args.output_json, 'w'))

