.PHONY: test format

test:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

format:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); using JuliaFormatter; format(["src","test","docs/examples"])'
