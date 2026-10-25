FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir "pytest>=8.0,<9" "ruff>=0.6,<1" "mypy>=1.10,<2"

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pytest -q

ENTRYPOINT ["python", "-m"]
