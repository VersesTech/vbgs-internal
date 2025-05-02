FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -ms /bin/bash intsys
ENV PYENV_ROOT="/home/intsys/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
WORKDIR /home/intsys/vbgs-internal

RUN chown -R intsys:intsys /home/intsys/vbgs-internal

RUN apt-get update && apt-get install -y curl build-essential libssl-dev zlib1g-dev\
	libbz2-dev libreadline-dev libsqlite3-dev git \
	libncursesw5-dev xz-utils libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev unzip \
	wget

USER intsys
RUN curl -fsSL https://pyenv.run | bash
RUN pyenv install 3.11.6 && pyenv global 3.11.6
COPY --chown=intsys pyproject.toml ./pyproject.toml
COPY --chown=intsys vbgs ./vbgs
COPY --chown=intsys scripts ./scripts
COPY --chown=intsys setup_docker.sh ./setup_docker.sh

RUN ./setup_docker.sh
ENV PYENV_VERSION=venv



