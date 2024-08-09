all: lint all

isort:
	poetry run isort --diff -c *.py
	poetry run isort --diff -c src/

black:
	poetry run python3 -m black --check *.py
	poetry run python3 -m black --check src/

flake:
	poetry run python3 -m flake8 *.py
	poetry run python3 -m flake8 src/

mypy:
	poetry run mypy --version
	poetry run mypy *.py
	poetry run mypy -p sparta.absa

lint: isort black flake mypy

format:
	poetry run isort *.py
	poetry run isort src/
	poetry run python3 -m black src/
