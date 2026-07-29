# Makefile with common project tasks

.PHONY: setup scrape inspect build eval viz test verify clean

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

scrape:
	python scrape_articles.py

inspect:
	python inspect_data.py

build:
	python build_pipelines.py

eval:
	python evaluate.py

viz:
	python visualize_results.py

test:
	pytest

verify:
	python verify_setup.py

clean:
	rm -rf __pycache__ .pytest_cache .cache chroma_db reports
	rm -f data.json
