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

import argparse

def add_neurosat_options(parser):
    parser.add_argument('--d', action='store', dest='d', type=int, default=128, help='Dimension of variable and clause embeddings')
    parser.add_argument('--n_rounds', action='store', dest='n_rounds', type=int, default=16, help='Number of rounds of message passing')

    parser.add_argument('--lr_decay_type', action='store', dest='lr_decay_type', type=str, default="exp")
    parser.add_argument('--lr_start', action='store', dest='lr_start', type=float, default=0.0001, help='Learning rate start')
    parser.add_argument('--lr_end', action='store', dest='lr_end', type=float, default=0.000001, help='Learning rate end')
    parser.add_argument('--lr_decay', action='store', dest='lr_decay', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--lr_decay_steps', action='store', dest='lr_decay_steps', type=float, default=5, help='Learning rate steps decay')
    parser.add_argument('--lr_power', action='store', dest='lr_power', type=float, default=0.5, help='Learning rate decay power')

    parser.add_argument('--l2_weight', action='store', dest='l2_weight', type=float, default=0.000000001, help='L2 regularization weight')
    parser.add_argument('--clip_val', action='store', dest='clip_val', type=float, default=0.5, help='Clipping norm')

    parser.add_argument('--lstm_transfer_fn', action='store', dest='lstm_transfer_fn', type=str, default="relu", help='LSTM transfer function')
    parser.add_argument('--vote_transfer_fn', action='store', dest='mlp_transfer_fn', type=str, default="relu", help='MLP transfer function')

    parser.add_argument('--final_reducer', action='store', dest='final_reducer', type=str, default="mean", help="Reducer for literal votes")

    parser.add_argument('--n_msg_layers', action='store', dest='n_msg_layers', type=int, default=3, help='Number of layers in message MLPs')
    parser.add_argument('--n_vote_layers', action='store', dest='n_vote_layers', type=int, default=3, help='Number of layers in vote MLPs')

    parser.add_argument('--tf_seed', action='store', dest='tf_seed', type=int, default=0, help='Random seed for tensorflow')
    parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=0, help='Random seed for numpy')
