# Convenience targets. `make help` lists them.
.PHONY: help check install install-cpu install-full test mock split demo abmil clam clean

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## install dependencies (GPU/default torch)
	pip install -r requirements.txt

install-cpu:  ## install dependencies with CPU-only torch
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

install-full:  ## install optional extras (streamlit, notebooks, torchvision, openslide)
	pip install -r requirements-full.txt

check:  ## verify the Python environment / package versions
	python scripts/check_env.py

test:  ## run the unit + end-to-end tests
	MPLBACKEND=Agg python -m unittest discover -s tests -v

mock:  ## full Person 2 pipeline on synthetic data (~1 min)
	bash scripts/run_mock_demo.sh

split:  ## create the stratified 300/60/60 split (needs data/raw/train.csv)
	python scripts/create_split.py --panda-csv data/raw/train.csv --out data/common_split.csv

clam:  ## CLAM-SB ablation + heatmaps + comparison on real features
	bash scripts/run_person2_experiments.sh

abmil:  ## ABMIL baseline (Person 1)
	python -m src.train --model abmil --config configs/abmil.yaml --out-dir results/metrics/abmil

demo:  ## launch the Streamlit demo
	streamlit run app/streamlit_app.py

clean:  ## remove generated synthetic data and mock results
	rm -rf data/mock results/mock_*
