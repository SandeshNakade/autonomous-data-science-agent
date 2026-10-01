FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --gid 10001 autods \
    && useradd --uid 10001 --gid autods --create-home --shell /usr/sbin/nologin autods

# Install the package before copying mutable runtime directories. The project
# metadata declares the FastAPI, Streamlit, ML, DVC, and monitoring dependencies.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[dashboard,mlops,boosting]"

COPY --chown=autods:autods . .
RUN mkdir -p /app/datasets /app/artifacts /app/models \
    && chown -R autods:autods /app

USER autods

EXPOSE 8000 8501

# The Streamlit service overrides this command in docker-compose.yml.
CMD ["uvicorn", "autods.api:app", "--host", "0.0.0.0", "--port", "8000"]
