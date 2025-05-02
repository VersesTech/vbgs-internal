#!/bin/bash

set -e

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed. Please install pyenv first."
    exit 1
fi

eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if ! pyenv versions | grep -q "3.11"; then
    echo "Installing Python 3.11 using pyenv..."
    pyenv install 3.11
fi
pyenv virtualenv 3.11 venv
export PYENV_VERSION=venv

pip install .

