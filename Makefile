.PHONY: build
build:
	poetry build

.PHONY: install
install:
	poetry install

.PHONY: clean
clean:
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} +

