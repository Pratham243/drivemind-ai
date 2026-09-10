.PHONY: setup data validate train-baseline train-transformer evaluate test lint format api dashboard docker-up docker-down

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	python scripts/generate_dataset.py

validate:
	python scripts/validate_dataset.py

train-baseline:
	python scripts/train_baseline.py

train-transformer:
	python scripts/train_transformer.py

evaluate:
	python scripts/evaluate.py

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/

api:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run app/streamlit_app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
