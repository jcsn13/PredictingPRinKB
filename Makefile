.PHONY: docs clean

VIRTUALENV_DIR=.env
PYTHON=${VIRTUALENV_DIR}/bin/python
PIP=${VIRTUALENV_DIR}/bin/pip

all:
	pip3 install virtualenv
	virtualenv -p python3 $(VIRTUALENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt	