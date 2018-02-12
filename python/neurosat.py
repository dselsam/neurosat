# Copyright 2018 Daniel Selsam. All Rights Reserved.
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

import tensorflow as tf
import numpy as np
import math
import random
import os
import time
from confusion import ConfusionMatrix
from problems_loader import init_problems_loader
from mlp import MLP
from util import repeat_end, decode_final_reducer, decode_transfer_fn
from tensorflow.contrib.rnn import LSTMStateTuple
from sklearn.cluster import KMeans

class NeuroSAT(object):
    def __init__(self, opts):
        self.opts = opts

        self.final_reducer = decode_final_reducer(opts.final_reducer)

        self.build_network()
        self.train_problems_loader = None

    def init_random_seeds(self):
        tf.set_random_seed(self.opts.tf_seed)
        np.random.seed(self.opts.np_seed)

    def construct_session(self):
        self.sess = tf.Session()

    def declare_parameters(self):
        opts = self.opts
        with tf.variable_scope('params') as scope:
            self.L_init = tf.get_variable(name="L_init", initializer=tf.random_normal([1, self.opts.d]))
            self.C_init = tf.get_variable(name="C_init", initializer=tf.random_normal([1, self.opts.d]))

            self.LC_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("LC_msg"))
            self.CL_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("CL_msg"))

            self.L_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.C_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))

            self.L_vote = MLP(opts, opts.d, repeat_end(opts.d, opts.n_vote_layers, 1), name=("L_vote"))
            self.vote_bias = tf.get_variable(name="vote_bias", shape=[], initializer=tf.zeros_initializer())

    def declare_placeholders(self):
        self.n_vars = tf.placeholder(tf.int32, shape=[], name='n_vars')
        self.n_lits = tf.placeholder(tf.int32, shape=[], name='n_lits')
        self.n_clauses = tf.placeholder(tf.int32, shape=[], name='n_clauses')

        self.L_unpack = tf.sparse_placeholder(tf.float32, shape=[None, None], name='L_unpack')
        self.is_sat = tf.placeholder(tf.bool, shape=[None], name='is_sat')

        # useful helpers
        self.n_batches = tf.shape(self.is_sat)[0]
        self.n_vars_per_batch = tf.div(self.n_vars, self.n_batches)

    def while_cond(self, i, L_state, C_state):
        return tf.less(i, self.opts.n_rounds)

    def flip(self, lits):
        return tf.concat([lits[self.n_vars:(2*self.n_vars), :], lits[0:self.n_vars, :]], axis=0)

    def while_body(self, i, L_state, C_state):
        LC_pre_msgs = self.LC_msg.forward(L_state.h)
        LC_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, LC_pre_msgs, adjoint_a=True)

        with tf.variable_scope('C_update') as scope:
            _, C_state = self.C_update(inputs=LC_msgs, state=C_state)

        CL_pre_msgs = self.CL_msg.forward(C_state.h)
        CL_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, CL_pre_msgs)

        with tf.variable_scope('L_update') as scope:
            _, L_state = self.L_update(inputs=tf.concat([CL_msgs, self.flip(L_state.h)], axis=1), state=L_state)

        return i+1, L_state, C_state

    def pass_messages(self):
        with tf.name_scope('pass_messages') as scope:
            denom = tf.sqrt(tf.cast(self.opts.d, tf.float32))

            L_output = tf.tile(tf.div(self.L_init, denom), [self.n_lits, 1])
            C_output = tf.tile(tf.div(self.C_init, denom), [self.n_clauses, 1])

            L_state = LSTMStateTuple(h=L_output, c=tf.zeros([self.n_lits, self.opts.d]))
            C_state = LSTMStateTuple(h=C_output, c=tf.zeros([self.n_clauses, self.opts.d]))

            _, L_state, C_state = tf.while_loop(self.while_cond, self.while_body, [0, L_state, C_state])

        self.final_lits = L_state.h
        self.final_clauses = C_state.h

    def compute_logits(self):
        with tf.name_scope('compute_logits') as scope:
            self.all_votes = self.L_vote.forward(self.final_lits) # n_lits x 1
            self.all_votes_join = tf.concat([self.all_votes[0:self.n_vars], self.all_votes[self.n_vars:self.n_lits]], axis=1) # n_vars x 2

            self.all_votes_batched = tf.reshape(self.all_votes_join, [self.n_batches, self.n_vars_per_batch, 2])
            self.logits = self.final_reducer(self.all_votes_batched) + self.vote_bias

    def compute_cost(self):
        self.predict_costs = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.is_sat, tf.float32))
        self.predict_cost = tf.reduce_mean(self.predict_costs)

        with tf.name_scope('l2') as scope:
            l2_cost = tf.zeros([])
            for var in tf.trainable_variables():
                l2_cost += tf.nn.l2_loss(var)

        self.cost = tf.identity(self.predict_cost + self.opts.l2_weight * l2_cost, name="cost")

    def build_optimizer(self):
        opts = self.opts

        self.global_step = tf.get_variable("global_step", shape=[], initializer=tf.zeros_initializer(), trainable=False)

        if opts.lr_decay_type == "no_decay":
            self.learning_rate = tf.constant(opts.lr_start)
        elif opts.lr_decay_type == "poly":
            self.learning_rate = tf.train.polynomial_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_end, power=opts.lr_power)
        elif opts.lr_decay_type == "exp":
            self.learning_rate = tf.train.exponential_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_decay, staircase=False)
        else:
            raise Exception("lr_decay_type must be 'no_decay', 'poly' or 'exp'")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, self.opts.clip_val)
        self.apply_gradients = optimizer.apply_gradients(zip(gradients, variables), name='apply_gradients', global_step=self.global_step)

    def initialize_vars(self):
        tf.global_variables_initializer().run(session=self.sess)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.opts.n_saves_to_keep)
        if self.opts.run_id:
            self.save_dir = "snapshots/run%d" % self.opts.run_id
            self.save_prefix = "%s/snap" % self.save_dir

    def build_network(self):
        self.init_random_seeds()
        self.construct_session()
        self.declare_parameters()
        self.declare_placeholders()
        self.pass_messages()
        self.compute_logits()
        self.compute_cost()
        self.build_optimizer()
        self.initialize_vars()
        self.init_saver()

    def save(self, epoch):
        self.saver.save(self.sess, self.save_prefix, global_step=epoch)

    def restore(self):
        snapshot = "snapshots/run%d/snap-%d" % (self.opts.restore_id, self.opts.restore_epoch)
        self.saver.restore(self.sess, snapshot)

    def build_feed_dict(self, problem):
        d = {}
        d[self.n_vars] = problem.n_vars
        d[self.n_lits] = problem.n_lits
        d[self.n_clauses] = problem.n_clauses

        d[self.L_unpack] =  tf.SparseTensorValue(indices=problem.L_unpack_indices,
                                                 values=np.ones(problem.L_unpack_indices.shape[0]),
                                                 dense_shape=[problem.n_lits, problem.n_clauses])

        d[self.is_sat] = problem.is_sat
        return d

    def train_epoch(self, epoch):
        if self.train_problems_loader is None:
            self.train_problems_loader = init_problems_loader(self.opts.train_dir)

        epoch_start = time.clock()

        epoch_train_cost = 0.0
        epoch_train_mat = ConfusionMatrix()

        train_problems, train_filename = self.train_problems_loader.get_next()
        for problem in train_problems:
            d = self.build_feed_dict(problem)
            _, logits, cost = self.sess.run([self.apply_gradients, self.logits, self.cost], feed_dict=d)
            epoch_train_cost += cost
            epoch_train_mat.update(problem.is_sat, logits > 0)

        epoch_train_cost /= len(train_problems)
        epoch_train_mat = epoch_train_mat.get_percentages()
        epoch_end = time.clock()

        learning_rate = self.sess.run(self.learning_rate)
        self.save(epoch)

        return (train_filename, epoch_train_cost, epoch_train_mat, learning_rate, epoch_end - epoch_start)

    def test(self, test_data_dir):
        test_problems_loader = init_problems_loader(test_data_dir)
        results = []

        while test_problems_loader.has_next():
            test_problems, test_filename = test_problems_loader.get_next()

            epoch_test_cost = 0.0
            epoch_test_mat = ConfusionMatrix()

            for problem in test_problems:
                d = self.build_feed_dict(problem)
                logits, cost = self.sess.run([self.logits, self.cost], feed_dict=d)
                epoch_test_cost += cost
                epoch_test_mat.update(problem.is_sat, logits > 0)

            epoch_test_cost /= len(test_problems)
            epoch_test_mat = epoch_test_mat.get_percentages()

            results.append((test_filename, epoch_test_cost, epoch_test_mat))

        return results

    def find_solutions(self, problem):
        def flip_vlit(vlit):
            if vlit < problem.n_vars: return vlit + problem.n_vars
            else: return vlit - problem.n_vars

        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        d = self.build_feed_dict(problem)
        all_votes, final_lits, logits, costs = self.sess.run([self.all_votes, self.final_lits, self.logits, self.predict_costs], feed_dict=d)

        solutions = []
        for batch in range(len(problem.is_sat)):
            decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0])
            decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit))

            def reify(phi):
                xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
                              [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
                def one_of(a, b): return (a and (not b)) or (b and (not a))
                assert(all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]

            if self.solves(problem, batch, decode_cheap_A): solutions.append(reify(decode_cheap_A))
            elif self.solves(problem, batch, decode_cheap_B): solutions.append(reify(decode_cheap_B))
            else:

                L = np.reshape(final_lits, [2 * n_batches, n_vars_per_batch, self.opts.d])
                L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)

                kmeans = KMeans(n_clusters=2, random_state=0).fit(L)
                distances = kmeans.transform(L)
                scores = distances * distances

                def proj_vlit_flit(vlit):
                    if vlit < problem.n_vars: return vlit - batch * n_vars_per_batch
                    else:                     return ((vlit - problem.n_vars) - batch * n_vars_per_batch) + n_vars_per_batch

                def decode_kmeans_A(vlit):
                    return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
                        scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]

                decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))

                if self.solves(problem, batch, decode_kmeans_A): solutions.append(reify(decode_kmeans_A))
                elif self.solves(problem, batch, decode_kmeans_B): solutions.append(reify(decode_kmeans_B))
                else: solutions.append(None)

        return solutions

    def solves(self, problem, batch, phi):
        start_cell = sum(problem.n_cells_per_batch[0:batch])
        end_cell = start_cell + problem.n_cells_per_batch[batch]

        if start_cell == end_cell:
            # no clauses
            return 1.0

        current_clause = problem.L_unpack_indices[start_cell, 1]
        current_clause_satisfied = False

        for cell in range(start_cell, end_cell):
            next_clause = problem.L_unpack_indices[cell, 1]

            # the current clause is over, so we can tell if it was unsatisfied
            if next_clause != current_clause:
                if not current_clause_satisfied:
                    return False

                current_clause = next_clause
                current_clause_satisfied = False

            if not current_clause_satisfied:
                vlit = problem.L_unpack_indices[cell, 0]
                #print("[%d] %d" % (batch, vlit))
                if phi(vlit):
                    current_clause_satisfied = True

        # edge case: the very last clause has not been checked yet
        if not current_clause_satisfied: return False
        return True
