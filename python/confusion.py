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

class ConfusionMatrix(object):
    def __init__(self):
        self.ff = 0
        self.ft = 0
        self.tf = 0
        self.tt = 0

    def add(self, other):
        self.ff += other.ff
        self.ft += other.ft
        self.tf += other.tf
        self.tt += other.tt

    def update_one(self, actual, predicted):
        if (not actual) and (not predicted):
            self.ff += 1
        elif (not actual) and predicted:
            self.ft += 1
        elif actual and (not predicted):
            self.tf += 1
        else:
            assert(actual and predicted)
            self.tt += 1

    def update(self, actuals, predicteds):
        for i in range(len(actuals)):
            self.update_one(actuals[i], predicteds[i])

    def get_percentages(self):
        total = self.ff + self.ft + self.tf + self.tt
        assert(total > 0)
        matrix = ConfusionMatrix()
        matrix.ff = float(self.ff) / total
        matrix.ft = float(self.ft) / total
        matrix.tf = float(self.tf) / total
        matrix.tt = float(self.tt) / total
        return matrix

    def __str__(self):
        return str((self.ff, self.ft, self.tf, self.tt))

    def __repr__(self):
        return self.__str__()

class FloatConfusionMatrix(object):
    def __init__(self):
        self.ff = []
        self.ft = []
        self.tf = []
        self.tt = []

    def update_one(self, actual, predicted, logit):
        if (not actual) and (not predicted):
            self.ff.append(logit)
        elif (not actual) and predicted:
            self.ft.append(logit)
        elif actual and (not predicted):
            self.tf.append(logit)
        else:
            assert(actual and predicted)
            self.tt.append(logit)

    def update(self, actuals, predicteds, logits):
        for i in range(len(actuals)):
            self.update_one(actuals[i], predicteds[i], logits[i])

    def get_stats(self):
        matrix = ConfusionMatrix()
        matrix.ff = (np.mean(self.ff), np.std(self.ff)) if self.ff != [] else (0, 0)
        matrix.ft = (np.mean(self.ft), np.std(self.ft)) if self.ft != [] else (0, 0)
        matrix.tf = (np.mean(self.tf), np.std(self.tf)) if self.tf != [] else (0, 0)
        matrix.tt = (np.mean(self.tt), np.std(self.tt)) if self.tt != [] else (0, 0)
        return matrix

    def __str__(self):
        return str((self.ff, self.ft, self.tf, self.tt))

    def __repr__(self):
        return self.__str__()
