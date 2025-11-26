FROM ghcr.io/astral-sh/uv:latest
WORKDIR /app
ADD . /app
RUN uv sync --locked
CMD ["uv", "run", "main.py"]