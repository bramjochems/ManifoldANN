PYTHON ?= python3

.PHONY: test format benchmark

test:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

format:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); using JuliaFormatter; format(["src","test","docs/examples"])'

benchmark:
	cd benchmarking && $(PYTHON) benchmark.py $(ARGS)
