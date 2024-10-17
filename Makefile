#
#	Makefile
#	Linear-Regression
#
#	Created by Marc MOSCA on 15/10/2024.
#

.DEFAULT_GOAL := help

.PHONY: bonus
bonus: ## Executes the bonus training program with "./data/dataset.csv" file.
	$(MAKE) install
	python3 src/main.py --bonus ./data/dataset.csv

.PHONY: help
help: ## Displays this help menu of commands available for the project.
	@echo "Available commands:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Installs the python dependencies needed for the project.
	python3 -m pip install -r requirements.txt

.PHONY: prediction
prediction: ## Executes the prediction program.
	$(MAKE) install
	python3 src/main.py --prediction

.PHONY: training
training: ## Executes the training program with "./data/dataset.csv" file.
	$(MAKE) install
	python3 src/main.py --training ./data/dataset.csv
