# BSD 3-Clause License

# Copyright (c) 2021, Juan L Gamella
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SUITE = all
PROJECT = utlvce

# Run tests
tests: test examples doctests

test:
ifeq ($(SUITE),all)
	python -m unittest discover $(PROJECT).test
else
	python -m unittest $(PROJECT).test.$(SUITE)
endif

# Run the example scripts in the README
examples:
	PYTHONPATH=./ python3 docs/algorithms_example.py
#	PYTHONPATH=./ python3 docs/equivalence_class_example.py
#	PYTHONPATH=./ python3 docs/equivalence_class_ges_example.py

# Run the doctests
doctests:
	PYTHONPATH=./ python3 $(PROJECT)/main.py
	PYTHONPATH=./ python3 $(PROJECT)/score.py
	PYTHONPATH=./ python3 $(PROJECT)/model.py
	PYTHONPATH=./ python3 $(PROJECT)/generators.py

# Set up virtual environment for execution
venv:
	python3 -m venv ./venv
	( \
	. venv/bin/activate; \
	pip install --upgrade pip setuptools; \
	pip install -r requirements.txt; \
	)
# Set up virtual environment for tests 
venv-tests:
	python3 -m venv ./venv-tests
	( \
	. venv-tests/bin/activate; \
	pip install --upgrade pip setuptools; \
	pip install -r requirements_tests.txt; \
	)


clean:
	rm -rf venv venv-tests

.PHONY: test, tests, examples, venv
