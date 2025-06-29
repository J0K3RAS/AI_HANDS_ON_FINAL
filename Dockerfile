FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y curl git
RUN mkdir /AI_HANDS_ON_FINAL
WORKDIR /AI_HANDS_ON_FINAL
RUN git clone -b main https://github.com/J0K3RAS/AI_HANDS_ON_FINAL.git .
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ~/.local/bin/uv python install cpython-3.12.9-linux-x86_64-gnu \
    && ~/.local/bin/uv lock  \
    && ~/.local/bin/uv sync
EXPOSE 8080
CMD ["/bin/bash", "-c", "source .venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8080"]
