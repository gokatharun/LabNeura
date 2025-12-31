# Write code to install LabNeura in python and test it
install:
	cd python && rm -rf build/ && pip install .
test: install
	python tests/test_correctness_large.py
	python tests/benchmark_numpy.py