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

import math
import os
import numpy as np
import tensorflow as tf
import random
import pickle
import argparse
import sys
from solver import solve_sat
from mk_problem import mk_batch_problem

def parse_dimacs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while lines[i].strip().split(" ")[0] == "c":
        i += 1
    header = lines[i].strip().split(" ")
    assert(header[0] == "p")
    n_vars = int(header[2])
    iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i+1:]]
    return n_vars, iclauses

def mk_dataset_filename(opts, n_batches):
    dimacs_path = opts.dimacs_dir.split("/")
    dimacs_dir = dimacs_path[-1] if dimacs_path[-1] != "" else dimacs_path[-2]
    return "%s/data_dir=%s_npb=%d_nb=%d.pkl" % (opts.out_dir, dimacs_dir, opts.max_nodes_per_batch, n_batches)

parser = argparse.ArgumentParser()
parser.add_argument('dimacs_dir', action='store', type=str)
parser.add_argument('out_dir', action='store', type=str)
parser.add_argument('max_nodes_per_batch', action='store', type=int)
parser.add_argument('--one', action='store', dest='one', type=int, default=0)
parser.add_argument('--max_dimacs', action='store', dest='max_dimacs', type=int, default=None)

opts = parser.parse_args()

problems = []
batches = []
n_nodes_in_batch = 0

filenames = os.listdir(opts.dimacs_dir)

if not (opts.max_dimacs is None):
    filenames = filenames[:opts.max_dimacs]

# to improve batching
filenames = sorted(filenames)

prev_n_vars = None

for filename in filenames:
    n_vars, iclauses = parse_dimacs("%s/%s" % (opts.dimacs_dir, filename))
    n_clauses = len(iclauses)
    n_cells = sum([len(iclause) for iclause in iclauses])

    n_nodes = 2 * n_vars + n_clauses
    if n_nodes > opts.max_nodes_per_batch:
        continue

    batch_ready = False
    if (opts.one and len(problems) > 0):
        batch_ready = True
    elif (prev_n_vars and n_vars != prev_n_vars):
        batch_ready = True
    elif (not opts.one) and n_nodes_in_batch + n_nodes > opts.max_nodes_per_batch:
        batch_ready = True

    if batch_ready:
        batches.append(mk_batch_problem(problems))
        print("batch %d done (%d vars, %d problems)...\n" % (len(batches), prev_n_vars, len(problems)))
        del problems[:]
        n_nodes_in_batch = 0

    prev_n_vars = n_vars

    is_sat, stats = solve_sat(n_vars, iclauses)
    problems.append((filename, n_vars, iclauses, is_sat))
    n_nodes_in_batch += n_nodes

if len(problems) > 0:
    batches.append(mk_batch_problem(problems))
    print("batch %d done (%d vars, %d problems)...\n" % (len(batches), n_vars, len(problems)))
    del problems[:]

# create directory
if not os.path.exists(opts.out_dir):
    os.mkdir(opts.out_dir)

dataset_filename = mk_dataset_filename(opts, len(batches))
print("Writing %d batches to %s...\n" % (len(batches), dataset_filename))
with open(dataset_filename, 'wb') as f_dump:
    pickle.dump(batches, f_dump)
