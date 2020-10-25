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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from opennmt.layers import transformer
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from pairwise_losses import pairwise_hinge_loss
# from slim.resnet152_img import extract_feature
from slim.nets.resnet_v1 import resnet_v1_152, resnet_arg_scope

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

class SummarizationModel(object):
    """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [FLAGS.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, None], name='enc_padding_mask')

        self._side_batch = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.max_side_steps, 32, 64, 3], name='side_batch')
        self._side_lens = tf.placeholder(tf.int32, [FLAGS.batch_size], name='side_lens')
        self._side_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.max_side_steps], name='side_padding_mask')
        self._segment_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, None], name='segment_padding_mask')

        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None],
                                                          name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')


        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps],
                                                name='dec_padding_mask')
        # self._dec_pic_target = tf.placeholder(tf.float32, [FLAGS.batch_size, None],
        #                                         name='target_pic_batch')
        self._dec_pic_target = tf.placeholder(tf.int32, [FLAGS.batch_size],
                                              name='target_pic_batch')

        if hps.mode == "decode" and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask

        feed_dict[self._side_batch] = batch.side_batch
        feed_dict[self._side_lens] = batch.side_lens
        feed_dict[self._side_padding_mask] = batch.side_padding_mask
        feed_dict[self._segment_padding_mask] = batch.segment_padding_mask

        if FLAGS.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
            feed_dict[self._dec_pic_target] = batch.dec_pic_target
        return feed_dict


    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=seq_len,
                                                                                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states

        return encoder_outputs, fw_st, bw_st

    def _add_side_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('side_encoder'):
            import opennmt as onmt
            fusion_self_encoder = onmt.encoders.SelfAttentionEncoder(2, num_units=2 * FLAGS.hidden_dim, num_heads=8)
            transformer_encoder_outputs, final_state, _ = fusion_self_encoder.encode(encoder_inputs, seq_len)
            transformer_fw_st = tf.contrib.rnn.LSTMStateTuple(final_state[0], final_state[0])
            transformer_bw_st = tf.contrib.rnn.LSTMStateTuple(final_state[1], final_state[1])
            encoder_outputs = transformer_encoder_outputs
            transformer_encoder_outputs.set_shape([FLAGS.batch_size, None, 2 * FLAGS.hidden_dim])
            fw_st = transformer_fw_st
            bw_st = transformer_bw_st
        return encoder_outputs, fw_st, bw_st


    def _add_side_rnn_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('side_rnn_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (rnn_encoder_outputs, (rnn_fw_st, rnn_bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=seq_len,
                                                                                swap_memory=True)
            rnn_encoder_outputs = tf.concat(axis=2, values=rnn_encoder_outputs)  # concatenate the forwards and backwards states
            # encoder_outputs = rnn_encoder_outputs
            fw_st = rnn_fw_st
            bw_st = rnn_bw_st
            side_states = tf.layers.dense(tf.concat(self._reduce_states(fw_st, bw_st), -1), FLAGS.hidden_dim*2)
            return side_states


    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            input_dim = fw_st.c.get_shape()[-1]
            w_reduce_c = tf.get_variable('w_reduce_c', [input_dim*2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [input_dim*2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)   # Get new cell from old cell
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def _add_decoder(self, inputs):
        """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

        Args:
          inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

        Returns:
          outputs: List of tensors; the outputs of the decoder
          out_state: The final state of the decoder
          attn_dists: A list of tensors; the attention distributions
          p_gens: A list of scalar tensors; the generation probabilities
          coverage: A tensor, the current coverage vector
        """
        hps = self._hps
        cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        prev_coverage = self.prev_coverage if hps.mode == "decode" and hps.coverage else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

        outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state,
                                                                             self._enc_states, self._enc_padding_mask,
                                                                             cell, initial_state_attention=(
                    hps.mode == "decode" or hps.mode == 'auto_decode'), pointer_gen=hps.pointer_gen,
                                                                             use_coverage=hps.coverage,
                                                                             prev_coverage=prev_coverage)

        return outputs, out_state, attn_dists, p_gens, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists):
        """Calculate the final distribution, for the pointer-generator model
            Args:
              vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
              attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
            Returns:
              final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
            """
        with tf.variable_scope('final_distribution'):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
            extended_vsize = self._vocab.size() + self._max_art_oovs  # the maximum (over the batch) size of the extended vocabulary
            extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                                    vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

            # Project the values in the attention distributions onto the appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection
            batch_nums = tf.range(0, limit=self._hps.batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
            indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
            shape = [self._hps.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                                    attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

            # Add the vocab distributions and the copy distributions together to get the final distributions
            # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
            # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
            final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                           zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists


    def pic_attention(self, emb_side_inputs):
        """Calculate the context vector and attention distribution from the decoder state.

        Args:
          decoder_state: state of the decoder
          coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

        Returns:
          context_vector: weighted sum of encoder_states
          attn_dist: attention distribution
          coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
        """
        with tf.variable_scope("Attention"):
            attention_vec_size = FLAGS.hidden_dim * 2
            # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
            decoder_features = tf.layers.dense(self._last_state, attention_vec_size)  # shape (batch_size, attention_vec_size)
            decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                              1)  # reshape to (batch_size, 1, 1, attention_vec_size)

            def masked_attention(e, mask):
                """Take softmax of e then apply enc_padding_mask and re-normalize"""
                attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                attn_dist *= mask  # apply mask
                masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

            v_h = tf.get_variable("v_h", [attention_vec_size])
            W_h = tf.get_variable("W_h", [1, 1, attention_vec_size, attention_vec_size])
            W_s = tf.get_variable("W_s", [1, 1, FLAGS.hidden_dim, attention_vec_size])

            side_features = tf.expand_dims(emb_side_inputs, axis=2)
            # segment_features = tf.expand_dims(self._side_states, axis=2)

            # segment_features = nn_ops.conv2d(segment_features, W_h, [1, 1, 1, 1], "SAME")
            side_features = nn_ops.conv2d(side_features, W_s, [1, 1, 1, 1], "SAME")

            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            # e_h = math_ops.reduce_sum(v_h * math_ops.tanh(segment_features + decoder_features),
            #                           [2, 3])  # calculate e
            e_s = math_ops.reduce_sum(v_h * math_ops.tanh(side_features + decoder_features),
                                      [2, 3])  # calculate e

            # Calculate attention distribution
            # attn_segment = masked_attention(e_h, self._segment_padding_mask)
            # attn_segment = tf.reshape(tf.tile(tf.expand_dims(attn_segment, 1), [1, 5, 1]), [FLAGS.batch_size, -1])
            attn_side = masked_attention(e_s, self._side_padding_mask)
            # attn_side = nn_ops.softmax(tf.multiply(attn_side, attn_segment))
            # attn_side = tf.multiply(attn_side, attn_segment)

        return attn_side

    def cross_attention(self, table_encodes, document_encodes, num_units, num_heads, num_layers, ffn_inner_dim,
                        sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
        table_encodes *= num_units ** 0.5
        # if self.position_encoder is not None:
        #     inputs = self.position_encoder(inputs)

        inputs = tf.layers.dropout(
            table_encodes,
            rate=FLAGS.attention_dropout,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        mask = transformer.build_sequence_mask(
            sequence_length,
            num_heads=num_heads,
            maximum_length=tf.shape(document_encodes)[1])

        state = ()

        for l in range(num_layers):
            with tf.variable_scope("layer_{}".format(l)):
                with tf.variable_scope("multi_head"):
                    context = transformer.multi_head_attention(
                        num_heads,
                        transformer.norm(inputs),
                        document_encodes,
                        mode,
                        num_units=num_units,
                        mask=mask,
                        dropout=FLAGS.attention_dropout)
                    context = transformer.drop_and_add(
                        inputs,
                        context,
                        mode,
                        dropout=FLAGS.dropout)

                with tf.variable_scope("ffn"):
                    transformed = transformer.feed_forward(
                        transformer.norm(context),
                        ffn_inner_dim,
                        mode,
                        dropout=FLAGS.attention_dropout)
                    transformed = transformer.drop_and_add(
                        context,
                        transformed,
                        mode,
                        dropout=FLAGS.dropout)

                inputs = transformed
                state += (tf.reduce_mean(inputs, axis=1),)

        outputs = transformer.norm(inputs)
        # return (outputs, state, sequence_length)
        return outputs


    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize = self._vocab.size()  # size of the vocabulary
        # with tf.variable_scope('image_encoder'):
        self.reshaped_pix = tf.reshape(self._side_batch, [-1, 32, 64, 3])
        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v1_152(self.reshaped_pix, is_training=FLAGS.mode == 'train')
            # feat1 = end_points['resnet_v1_152/block4']
        pic_encoded = end_points['global_pool']
        # self.end_points = end_points
        # self.net = net

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding,
                                                        self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch,
                                                                                           axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)
            pic_encoded = tf.reshape(tf.squeeze(pic_encoded), [FLAGS.batch_size, FLAGS.max_side_steps, -1])
            emb_side_inputs = tf.layers.dense(pic_encoded, FLAGS.emb_dim * 2)
            # Add the encoder.
            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            # batch_size * pic_num * emb_dim
            new_emb_side_inputs = tf.reshape(emb_side_inputs, [FLAGS.batch_size * int(FLAGS.max_side_steps/5), 5, FLAGS.hidden_dim*2])
            # (batch_size*pic_num/5) * 5 * emb_dim

            side_states = self._add_side_rnn_encoder(new_emb_side_inputs, 5 * tf.ones((new_emb_side_inputs.get_shape()[0]), dtype=tf.int32))
            self._side_inputs = tf.reshape(side_states, [FLAGS.batch_size, -1, FLAGS.hidden_dim * 2])
            self._enc_states = enc_outputs

            # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
            self._dec_in_state = self._reduce_states(fw_st, bw_st)
            self._last_state = tf.concat(self._dec_in_state, -1)

            with tf.variable_scope('interaction'):
                change_side_states = tf.transpose(self._side_inputs, [0, 2, 1])
                self._change_side_states = change_side_states
                attn_matrix = tf.matmul(self._enc_states, change_side_states)
                # batch_size * enc_len * side_len
                self._video_aware_enc_states = tf.matmul(attn_matrix, self._side_inputs)
                self._news_aware_side_states = tf.matmul(tf.transpose(attn_matrix, [0, 2, 1]), self._enc_states)
                gate = tf.layers.dense(self._last_state, 1, activation=tf.nn.sigmoid)
                gate = tf.expand_dims(tf.tile(gate, [1, FLAGS.hidden_dim * 2]), 1)
                ones = np.ones([FLAGS.batch_size, 1, FLAGS.hidden_dim * 2])
                self._enc_states = gate * self._enc_states + (ones - gate) * self._video_aware_enc_states

            # Add the decoder.
            with tf.variable_scope('decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)
                # attn_seg, attn_side = self.pic_attention(emb_side_inputs)
                # self._attn_side = attn_side

            # Add the output projection to obtain the vocabulary distribution
            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                w_t = tf.transpose(w)
                v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v))  # apply the linear layer

                vocab_dists = [tf.nn.softmax(s) for s in
                               vocab_scores]  # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

            # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
            if FLAGS.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
            else:  # final distribution is just vocabulary distribution
                final_dists = vocab_dists

            if hps.mode in ['train', 'eval']:
                # Calculate the loss
                with tf.variable_scope('loss'):
                    if FLAGS.pointer_gen:
                        # Calculate the loss per step
                        # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                        loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                        batch_nums = tf.range(0, limit=hps.batch_size)  # shape (batch_size)
                        for dec_step, dist in enumerate(final_dists):
                            targets = self._target_batch[:, dec_step]  # The indices of the target words. shape (batch_size)
                            indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                            gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
                            losses = -tf.log(gold_probs + 1e-10)
                            loss_per_step.append(losses)

                        # Apply dec_padding_mask and get loss
                        self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                    else:  # baseline model
                        self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
                                                                      self._target_batch,
                                                                      self._dec_padding_mask)  # this applies softmax internally

                    tf.summary.scalar('loss', self._loss)

                    # Calculate coverage loss from the attention distributions
                    if hps.coverage:
                        with tf.variable_scope('coverage_loss'):
                            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                            tf.summary.scalar('coverage_loss', self._coverage_loss)
                        self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
                        tf.summary.scalar('total_loss', self._total_loss)

                # with tf.variable_scope('pic_loss'):
                #     self._loss_pic = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=attn_side,
                #                                                                        labels=self._dec_pic_target))
                #     # self._loss_unified = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=attn_side,
                #     #                                                                    labels=attn_seg))
                # self._all_loss = self._loss_pic
                # self._all_loss = self._loss

        with tf.variable_scope('side'):
            emb_side_inputs = tf.nn.l2_normalize(emb_side_inputs, dim=-1)

            # self-attention
            side_outputs, sfw_st, sbw_st = self._add_side_encoder(self._side_inputs, self._side_lens)
            conditional_vec = tf.expand_dims(self._last_state, 1)
            conditional_weight = tf.layers.dense(tf.multiply(conditional_vec, side_outputs), 1)
            self._cond_side_states = tf.multiply(side_outputs, conditional_weight)

            s_gate = tf.layers.dense(self._last_state, 1, activation=tf.nn.sigmoid)
            s_gate = tf.expand_dims(s_gate, 1)
            s_ones = np.ones_like(s_gate)
            self._side_states = s_gate * self._news_aware_side_states + (s_ones - s_gate) * self._cond_side_states

            fusion_gate = tf.layers.dense(self._last_state, 1, activation=tf.nn.sigmoid)
            fusion_gate = tf.expand_dims(tf.tile(fusion_gate, [1, FLAGS.hidden_dim * 2]), 1)
            fusion_ones = tf.ones_like(fusion_gate)
            side_states = tf.nn.l2_normalize(tf.reshape(tf.tile(tf.expand_dims(self._side_states, 1), [1, 5, 1, 1]), [FLAGS.batch_size, -1, FLAGS.hidden_dim * 2]), dim=-1)
            fusion_side = fusion_gate * emb_side_inputs + (fusion_ones - fusion_gate) * side_states

            attn_side = tf.squeeze(tf.layers.dense(fusion_side, 1, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            attn_side = nn_ops.softmax(attn_side)
            self.attn_side = attn_side

            # last_state = tf.nn.l2_normalize(tf.tile(tf.expand_dims(self._last_state, 1), [1, 10, 1]), dim=-1)
            # emb_side_inputs = tf.nn.l2_normalize(emb_side_inputs, dim=-1)
            # attn_side = tf.squeeze(tf.layers.dense(tf.concat([last_state, emb_side_inputs], -1), 1, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            # self.attn_side = attn_side

            with tf.variable_scope('pic_loss'):
                # self._loss_pic = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=attn_side,
                #                                                                         labels=self._dec_pic_target))
                # self._loss_pic = pairwise_hinge_loss(logits=attn_side, labels=self._dec_pic_target)
                self._loss_pic = pairwise_hinge_loss(logits=attn_side, labels=tf.one_hot(self._dec_pic_target, FLAGS.max_side_steps))
        if hps.mode in ['train', 'eval']:
            self._all_loss = self._loss + self._loss_pic



        if hps.mode == "decode" or hps.mode == 'auto_decode':
            # We run decode beam search mode one decoder step at a time
            assert len(final_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
            self._topk_log_probs = tf.log(topk_probs)

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        # loss_to_minimize = self._total_loss if FLAGS.coverage else self._loss
        # loss_to_minimize_1 = self._loss
        # tvars_1 = tf.trainable_variables(scope='seq2seq')
        # gradients_1 = tf.gradients(loss_to_minimize_1, tvars_1, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        loss_to_minimize_2 = self._all_loss
        tvars_2 = tf.trainable_variables()
        # tvars_2 = tf.trainable_variables(scope='side')
        gradients_2 = tf.gradients(loss_to_minimize_2, tvars_2, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)


        # # Clip the gradients
        # tf.logging.info('Clipping gradients...')
        # with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
        #     grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
        #     tf.summary.scalar('loss/global_norm', global_norm)
        #
        # learning_rate = tf.train.polynomial_decay(FLAGS.lr, self.global_step,
        #                                           FLAGS.dataset_size / FLAGS.batch_size * 5,
        #                                           FLAGS.lr / 10)
        # tf.summary.scalar('loss/learning_rate', learning_rate)
        # # Apply adagrad optimizer
        # if FLAGS.optimizer == 'adagrad':
        #     optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=FLAGS.adagrad_init_acc)
        # elif FLAGS.optimizer == 'adam':
        #     optimizer = tf.train.AdamOptimizer(learning_rate)
        # else:
        #     raise NotImplementedError()


        # Clip the gradients
        with tf.device("/gpu:0"):
            # grads_1, global_norm_1 = tf.clip_by_global_norm(gradients_1, self._hps.max_grad_norm)
            grads_2, global_norm_2 = tf.clip_by_global_norm(gradients_2, self._hps.max_grad_norm)

        # Add a summary
        # tf.summary.scalar('global_norm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdamOptimizer(self._hps.lr)
        # optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        with tf.device("/gpu:0"):
            # self._train_op_1 = optimizer.apply_gradients(zip(grads_1, tvars_1), global_step=self.global_step, name='train_step_1')
            self._train_op_2 = optimizer.apply_gradients(zip(grads_2, tvars_2), global_step=self.global_step, name='train_step_2')


    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        with tf.device("/gpu:0"):
            self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        # tf.logging.info(np.argmax(np.array(tmp_attn), axis=1))
        to_return = {
            # 'train_op_1': self._train_op_1,
            'train_op_2': self._train_op_2,
            # 'summaries': self._summaries,
            'all_loss': self._all_loss,
            's2s_loss': self._loss,
            'pic_loss': self._loss_pic,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
          sess: Tensorflow session.
          batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
          enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
          dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True)  # feed the batch into the placeholders
        (enc_states, dec_in_state, attn_side, global_step) = sess.run([self._enc_states, self._dec_in_state, self.attn_side, self.global_step],
                                                           feed_dict)  # run the encoder
        # tmp_attn = sess.run(self._attn_side, feed_dict)
        labels = np.argmax(np.array(attn_side), axis=1)
        # logits = batch.dec_pic_target
        # same = 0
        # for i in range(len(labels)):
        #     if labels[i] == logits[i]:
        #         same += 1
        # tf.logging.info(same / len(labels))
        # if labels[0] == logits[0]:
            # same += 1
            # tgt = open('/home1/lmz/video_data/test_multi', 'a', encoding='utf-8')
            # tgt.write(batch.original_abstracts[0] + '\n')
        # print(same)
        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        # pic_num = np.argmax(np.array(attn_side), 1)
        # pic = (batch.side_batch.tolist())[pic_num]
        return enc_states, dec_in_state, labels, attn_side

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
        """For beam search decoding. Run the decoder for one step.

        Args:
          sess: Tensorflow session.
          batch: Batch object containing single example repeated across the batch
          latest_tokens: Tokens to be fed as input into the decoder for this timestep
          enc_states: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }

        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(beam_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        if FLAGS.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss


def batch_coattention_nnsubmulti(utterance, response, utterance_mask, scope="co_attention", reuse=None):
    '''Point-wise interaction. (NNSUBMULTI)
    Args:
      utterance: [batch*turns, len_utt, dim]
      response: [batch*turns, len_res, dim]
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as response
    '''

    with tf.variable_scope(scope, reuse=reuse):
        dim = utterance.get_shape().as_list()[-1]

        weight = tf.get_variable('Weight', shape=[dim, dim], dtype=tf.float32)
        e_utterance = tf.einsum('aij,jk->aik', utterance, weight)
        # [batch, len_res, dim] * [batch, len_utterance, dim]T -> [batch, len_res, len_utterance]
        a_matrix = tf.matmul(response, tf.transpose(e_utterance, perm=[0, 2, 1]))  # [batch, len_res, len_utterance]

        # [batch, len_res, len_utterance] * [?, len_utterance, dim] -> [?, len_res, dim]
        # a_matrix = tf.add(a_matrix, tf.expand_dims(tf.multiply(tf.constant(-1000000.0), 1-utterance_mask), axis=1))
        # reponse_atten = tf.matmul(tf.nn.softmax(a_matrix, dim=-1), utterance)

        reponse_atten = tf.matmul(rx_masked_softmax(a_matrix, utterance_mask), utterance)  #

        feature_mul = tf.multiply(reponse_atten, response)
        feature_sub = tf.subtract(reponse_atten, response)
        feature_last = tf.layers.dense(tf.concat([feature_mul, feature_sub], axis=-1), dim, use_bias=True,
                                       activation=tf.nn.relu, reuse=reuse)  # [batch*turn, len, dim]
    return feature_last

def rx_masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.
    Input shape: (batch_size, len_res, len_utt).
    mask parameter: Tensor of shape (batch_size, len_utt). Such a mask is given by the length() function.
    return shape: [batch_size, len_res, len_utt]
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 2, keep_dims=True))) * tf.expand_dims(mask, axis=1)
    denominator = tf.reduce_sum(numerator, 2, keep_dims=True)

    # weights = tf.div(numerator, denominator)
    # weights = tf.div(numerator, denominator+1e-3)
    weights = tf.div(numerator + 1e-5 / mask.get_shape()[-1].value, denominator + 1e-5)
    return weights

