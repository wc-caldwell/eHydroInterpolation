FROM ghcr.io/prefix-dev/pixi@sha256:2a7e278d8e01cccdb1a9d851dd16804d0c7c41a82a39ebb07c2a1be25a0dd894

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pixi.toml pixi.lock ./

RUN pixi lock \
    && pixi install -a \
    && pixi clean cache --yes

COPY . .

ENTRYPOINT ["pixi", "run", "--"]
CMD ["python", "--version"]