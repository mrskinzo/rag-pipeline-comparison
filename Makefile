# Makefile with common project tasks

.PHONY: setup scrape inspect test eval clean

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

scrape:
	python scrape_articles.py

inspect:
	python inspect_data.py

eval:
	python evaluate.py

test:
	pytest

clean:
	rm -rf __pycache__
	rm -f evaluation_results.csv articles.json data.json
