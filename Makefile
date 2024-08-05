all: lint all

isort:
	poetry run isort --diff -c ft_absa_ddp.py
	poetry run isort --diff -c ft_absa_fsdp.py
	poetry run isort --diff -c src/

black:
	poetry run python3 -m black --check ft_absa_ddp.py
	poetry run python3 -m black --check ft_absa_fsdp.py
	poetry run python3 -m black --check src/

flake:
	poetry run python3 -m flake8 ft_absa_ddp.py
	poetry run python3 -m flake8 ft_absa_fsdp.py
	poetry run python3 -m flake8 src/

mypy:
	poetry run mypy --version
	poetry run mypy ft_absa_ddp.py
	poetry run mypy ft_absa_fsdp.py
	poetry run mypy -p sparta.absa

lint: isort black flake mypy

format:
	poetry run isort ft_absa_ddp.py
	poetry run isort ft_absa_fsdp.py
	poetry run isort src/
	poetry run python3 -m black src/
