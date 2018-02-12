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

import numpy as np
import math

# TODO(dhs): duplication
def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

# TODO(dhs): duplication
def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

class Problem(object):
    def __init__(self, n_vars, iclauses, is_sat, n_cells_per_batch, all_dimacs):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.n_clauses = len(iclauses)

        self.n_cells = sum(n_cells_per_batch)
        self.n_cells_per_batch = n_cells_per_batch

        self.is_sat = is_sat
        self.compute_L_unpack(iclauses)

        # will be a list of None for training problems
        self.dimacs = all_dimacs

    def compute_L_unpack(self, iclauses):
        self.L_unpack_indices = np.zeros([self.n_cells, 2], dtype=np.int)
        cell = 0
        for clause_idx, iclause in enumerate(iclauses):
            vlits = [ilit_to_vlit(x, self.n_vars) for x in iclause]
            for vlit in vlits:
                self.L_unpack_indices[cell, :] = [vlit, clause_idx]
                cell += 1

        assert(cell == self.n_cells)

def shift_ilit(x, offset):
    assert(x != 0)
    if x > 0: return x + offset
    else:     return x - offset

def shift_iclauses(iclauses, offset):
    return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]

def mk_batch_problem(problems):
    all_iclauses = []
    all_is_sat = []
    all_n_cells = []
    all_dimacs = []
    offset = 0

    prev_n_vars = None
    for dimacs, n_vars, iclauses, is_sat in problems:
        assert(prev_n_vars is None or n_vars == prev_n_vars)
        prev_n_vars = n_vars

        all_iclauses.extend(shift_iclauses(iclauses, offset))
        all_is_sat.append(is_sat)
        all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
        all_dimacs.append(dimacs)
        offset += n_vars

    return Problem(offset, all_iclauses, all_is_sat, all_n_cells, all_dimacs)
