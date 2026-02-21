.PHONY: install install-dev lint format test train serve producer consumer demo clean help

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v --cov=src

train:
	python -m src.models.trainer

evaluate:
	python -m src.models.evaluator

serve:
	uvicorn src.serving.app:app --reload --port 8000

producer:
	python -m src.streaming.producer --local --limit 100 --speed 10

consumer:
	python -m src.streaming.consumer --local --limit 100

demo:
	@echo "Starting demo pipeline..."
	python -m src.streaming.producer --local --limit 50 --speed 100
	python -m src.streaming.consumer --local --limit 50
	python -m src.monitoring.metrics_logger
	@echo "Demo complete! Check data/stream_output/ and data/monitoring/"

clean:
	rm -rf __pycache__ .pytest_cache mlruns
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f data/stream_output/*.jsonl data/monitoring/*.jsonl data/monitoring/*.log

help:
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install dev dependencies"
	@echo "  make lint         Run linter"
	@echo "  make format       Format code"
	@echo "  make test         Run tests"
	@echo "  make train        Train all models"
	@echo "  make evaluate     Run model evaluation"
	@echo "  make serve        Start FastAPI server"
	@echo "  make producer     Run transaction producer"
	@echo "  make consumer     Run transaction consumer"
	@echo "  make demo         Run full demo pipeline"
	@echo "  make clean        Remove caches and temp files"
