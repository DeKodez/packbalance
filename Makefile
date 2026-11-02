.PHONY: install test lint format data train bench profile clean

install:
	python -m pip install -e ".[data,dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

data:
	python -m packbalance.data.prepare --config configs/tiny.yaml

train:
	torchrun --standalone --nproc_per_node=$${NGPU:-1} -m packbalance.train.train --config configs/tiny.yaml

bench:
	python -m packbalance.profiling.bench_train --config configs/bench.yaml

profile:
	python -m packbalance.train.train --config configs/tiny.yaml --profile

clean:
	rm -rf artifacts checkpoints .pytest_cache .ruff_cache
