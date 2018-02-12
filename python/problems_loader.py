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

import os
import pickle
import mk_problem

class ProblemsLoader(object):
    def __init__(self, filenames):
        self.filenames = filenames
        print(self.filenames)

        self.next_file_num = 0
        assert(self.has_next())

    def has_next(self):
        return self.next_file_num < len(self.filenames)

    def get_next(self):
        if not self.has_next():
            self.reset()
        filename = self.filenames[self.next_file_num]
        print("Loading %s..." % filename)
        with open(filename, 'rb') as f:
            problems = pickle.load(f)
        self.next_file_num += 1
        assert(len(problems) > 0)
        return problems, filename

    def reset(self):
        self.next_file_num = 0

def init_problems_loader(dirname):
    return ProblemsLoader([dirname + "/" + f for f in os.listdir(dirname)])
