.PHONY: setup data validate train-baseline train-transformer evaluate test lint format api dashboard docker-up docker-down

# Note: creates the venv but does NOT install into it directly (the venv's
# pip lives at .venv/bin/pip on Linux/macOS vs .venv/Scripts/pip.exe on
# Windows). Activate the venv after this step, then run `pip install -r
# requirements.txt` — see README "Installation & local setup".
setup:
	python -m venv .venv

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
