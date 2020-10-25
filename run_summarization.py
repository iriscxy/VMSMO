# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import util
from tensorflow.python import debug as tf_debug
import subprocess
import shutil
import zipfile
import glob
from datetime import datetime
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

# Where to find data
tf.app.flags.DEFINE_string('data_path', '',
                           'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('test_path', '', 'path of test.'),
tf.app.flags.DEFINE_string('vocab_path', '/share/lmz/video_sum_data/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('pretrain_pic_ckpt', '/home1/lmz/video_data/resnet_v1_152.ckpt', 'sticker_path.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 128, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('max_side_steps', 10, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_enum('inception_endpoint', 'mixed_17x17x768e', ['mixed_8x8x2048b', 'mixed_17x17x768e'], '手工运行auto deocde不会有输出')


tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 10,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

tf.app.flags.DEFINE_integer('placeholder', 0, 'placeholder')
tf.app.flags.DEFINE_integer('epoch_num', None, 'placeholder')
tf.flags.DEFINE_string('current_source_code_zip', None, "current_source_code_zip")

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False,
                            'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0,
                          'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False,
                            'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')


def calc_running_avg_loss(loss, running_avg_loss, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
      loss: loss on the most recent eval step
      running_avg_loss: running_avg_loss so far
      step: training iteration step
      decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
      running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()  # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph()  # build the graph
    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    if FLAGS.restore_best_model:
        restore_best_model()
    # if not FLAGS.pretrain:
    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    epoch_saver = tf.train.Saver(max_to_keep=99)  # keep 3 checkpoints at a time
    checkpoint_exclude_scopes = 'seq2seq,side'
    exclusions = None
    if checkpoint_exclude_scopes:
        exclusions = [
            scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)
    pretrain_saver = tf.train.Saver(var_list=variables_to_restore)
    # pretrain_saver = tf.train.Saver(var_list={v.name[:-2]: v for v in tf.trainable_variables() if v.name.startswith('image_encoder') and 'logits' not in v.name})  # keep 3 checkpoints at a time
    # and 'logits' not in v.name
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=600,  # save summaries for tensorboard every 60 secs
                             save_model_secs=600,  # checkpoint every 60 secs
                             global_step=model.global_step)
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info("Created session.")
    tf.logging.info("Loading pretrained parameters.")
    pretrain_saver.restore(sess_context_manager, FLAGS.pretrain_pic_ckpt)
    try:
        run_training(model, batcher, sess_context_manager, sv,
                     epoch_saver)  # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()

    # else:
    #     # Initialize all vars in the model
    #     sess = tf.Session(config=util.get_config())
    #     print("Initializing all variables...")
    #     sess.run(tf.initialize_all_variables())
    #
    #     # Restore the best model from eval dir
    #     saver = tf.train.Saver(
    #         [v for v in tf.all_variables() if
    #          "Adagrad" not in v.name and "recognitionNetwork" not in v.name and "priorNetwork" not in v.name and "bow" not in v.name])
    #     print("Restoring all non-adagrad variables from best model in eval dir...")
    #     path = '/home1/chenxy/ep4'
    #     saver.restore(sess, path)
    #     print("Restored %s." % path)
    #     epoch_saver = tf.train.Saver(max_to_keep=99)  # keep 3 checkpoints at a time
    #     try:
    #         run_training(model, batcher, sess, saver,
    #                       epoch_saver)  # this is an infinite loop until interrupted
    #     except KeyboardInterrupt:
    #         tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    #         sess.stop()


def get_available_gpu():
    child = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,utilization.memory', '--format=csv,noheader'],
        stdout=subprocess.PIPE)
    text = str(child.communicate()[0], 'utf8')
    reader = csv.reader(text.split('\n')[:-1])
    stats = []
    for line in reader:
        memory = int(line[1].strip()[:-4])  # MB
        compute_percentage = int(line[2].strip()[:-1])  # %
        memory_percentage = int(line[3].strip()[:-1])  # %
        print(
            'index:{} memory:{} compute_percentage:{} memory_percentage:{}'.format(line[0], memory, compute_percentage,
                                                                                   memory_percentage))
        stats.append({'id': line[0], 'memory': memory})
        if memory < 9000:
            print('\033[32m gpu ' + line[0] + ' is available \033[0m')
            return int(line[0])


def start_auto_decode_proc(epoch_num=None):
    def run_command(command, stdout=None):
        if stdout is None:
            with open(os.devnull, 'w') as devnull:
                child = subprocess.Popen(command, shell=True, stdout=devnull)
                return child
        else:
            child = subprocess.Popen(command, shell=True, stdout=stdout)
            return child

    gpu_num = get_available_gpu()

    # 创建decode所需flags
    gpu_str = 'CUDA_VISIBLE_DEVICES=' + str(gpu_num)
    flag_str = ''
    except_key = ['mode', 'data_path', 'test_path', 'log_root', 'h', 'help', 'helpfull', 'helpshort', 'current_source_code_zip',
                  'single_pass', 'placeholder']
    for key, val in FLAGS.__flags.items():
        val = val._value
        if key not in except_key and val is not None:
            flag_str += '--%s=%s ' % (key, val)
        elif key == 'mode':
            flag_str += '--mode=auto_decode '
        elif key == 'single_pass':
            flag_str += '--single_pass=True '
        elif key == 'placeholder':
            flag_str += '--placeholder=0 '
        elif key == 'test_path':
            flag_str += '--data_path=%s ' % val
        elif key == 'log_root':
            flag_str += '--log_root=%s ' % os.path.abspath(os.path.join(FLAGS.log_root, '../'))
    flag_str += '--epoch_num=' + str(epoch_num)

    # 解压train code压缩包
    source_code_path = os.path.join(os.path.abspath(os.path.dirname(FLAGS.current_source_code_zip)), 'train_code')
    if os.path.exists(source_code_path):
        shutil.rmtree(source_code_path)
    zip_ref = zipfile.ZipFile(FLAGS.current_source_code_zip, 'r')
    zip_ref.extractall(source_code_path)
    zip_ref.close()
    tf.logging.info('unzip source code finish!')

    run_file_path = os.path.join(source_code_path, 'run_summarization.py')
    tf.logging.debug(' '.join([gpu_str, sys.executable, run_file_path, flag_str]))
    child = run_command(' '.join([gpu_str, sys.executable, run_file_path, flag_str]))
    sys.stderr.write(' '.join([gpu_str, sys.executable, run_file_path, flag_str]) + '\n')
    sys.stderr.flush()


def run_training(model, batcher, sess_context_manager, sv,  epoch_saver):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tf.logging.info("starting run_training")

    def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            # pylint: disable=W0212
            session = session._sess
        return session

    with sess_context_manager as sess:
        losses = []
        while True:  # repeats until interrupted
            batch = batcher.next_batch()
            results = model.run_train_step(sess, batch)
            train_step = results['global_step']  # we need this to update our running average loss
            loss = results['all_loss']
            losses.append(loss)

            if train_step % 100 == 0:
                tf.logging.info('train_step: %f', train_step)  # print the loss to screen
                tf.logging.info('loss: %f', sum(losses)/100)  # print the loss to screen
                tf.logging.info('s2s_loss: %f', results['s2s_loss'])  # print the loss to screen
                tf.logging.info('pic_loss: %f', results['pic_loss'])  # print the loss to screen
                tf.logging.info('===================')  # print the loss to screen
                losses = []

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            if FLAGS.coverage:
                coverage_loss = results['coverage_loss']
                tf.logging.info("coverage_loss: %f", coverage_loss)  # print the coverage loss to screen

            if train_step % 12500 == 0:
                print('saving')
                epoch_num = int(train_step / 12500)
                epoch_ckpt_dir = os.path.join(FLAGS.log_root, "epoch_ckpt")
                epoch_saver.save(get_session(sess), os.path.join(epoch_ckpt_dir, 'ep{}'.format(epoch_num)))
                print('start auto decode for epoch {}'.format(epoch_num))
                start_auto_decode_proc(epoch_num)


def run_eval(model, batcher, vocab):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    model.build_graph()  # build the graph
    saver = tf.train.Saver(max_to_keep=10)  # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval")  # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')  # this is where checkpoints of best models are saved
    running_avg_loss = 0  # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess)  # load a new checkpoint
        batch = batcher.next_batch()  # get the next batch

        # run eval on the batch
        t0 = time.time()
        results = model.run_eval_step(sess, batch)
        t1 = time.time()
        tf.logging.info('seconds for batch: %.2f', t1 - t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        tf.logging.info('loss: %f', loss)
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)

        # add summaries
        train_step = results['global_step']

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss,
                            bestmodel_save_path)
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss


def main(unused_argv):
    if FLAGS.placeholder:
        tf.logging.info('try to occupy GPU memory!')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        placeholder_session = tf.Session(config=config)
        limit = placeholder_session.run(tf.contrib.memory_stats.BytesLimit()) / 1073741824
        tf.logging.info('occupy GPU memory %f GB', limit)
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode' or FLAGS.mode == 'auto_decode':
        FLAGS.batch_size = FLAGS.beam_size

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'max_side_steps', 'coverage', 'cov_loss_wt',
                   'pointer_gen', 'epoch_num', 'current_source_code_zip', 'multi_dec_steps']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val.value  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # save python source code
    current_time_str = datetime.now().strftime('%m-%d-%H-%M')
    FLAGS.current_source_code_zip = os.path.abspath(
        os.path.join(FLAGS.log_root, 'source_code_bak-' + current_time_str + '-' + FLAGS.mode + '.zip'))
    tf.logging.info('saving source code: %s', FLAGS.current_source_code_zip)
    python_list = glob.glob('./*.py')
    zip_file = zipfile.ZipFile(FLAGS.current_source_code_zip, 'w')
    for d in python_list:
        zip_file.write(d)
    zip_file.close()

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111)  # a seed value for randomness

    if hps.mode == 'train':
        print("creating model...")
        model = SummarizationModel(hps, vocab)
        if FLAGS.placeholder:
            placeholder_session.close()
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps  # This will be the hyperparameters for the decoder model
        decode_model_hps = hps._replace(
            max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder.decode()  # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
    elif hps.mode == 'auto_decode':
        decode_model_hps = hps  # This will be the hyperparameters for the decoder model
        decode_model_hps = hps._replace(
            max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab, hps.epoch_num)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
    tf.app.run()
