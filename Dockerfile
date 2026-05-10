FROM nvcr.io/nvidia/pytorch:25.02-py3

ARG NODE_VERSION=24.15.0
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    openssh-server \
    curl \
    xz-utils \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd /usr/local/lib/nodejs

RUN curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
 && tar -xJf "node-v${NODE_VERSION}-linux-x64.tar.xz" -C /usr/local/lib/nodejs \
 && ln -sf "/usr/local/lib/nodejs/node-v${NODE_VERSION}-linux-x64/bin/node" /usr/local/bin/node \
 && ln -sf "/usr/local/lib/nodejs/node-v${NODE_VERSION}-linux-x64/bin/npm" /usr/local/bin/npm \
 && ln -sf "/usr/local/lib/nodejs/node-v${NODE_VERSION}-linux-x64/bin/npx" /usr/local/bin/npx \
 && ln -sf "/usr/local/lib/nodejs/node-v${NODE_VERSION}-linux-x64/bin/corepack" /usr/local/bin/corepack \
 && rm -f "node-v${NODE_VERSION}-linux-x64.tar.xz"

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" \
    -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

WORKDIR /workspace

EXPOSE 22 30869 8889

CMD ["/bin/bash"]
