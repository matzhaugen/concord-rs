dev: # start dev python environment
	source .venv/bin/activate && pip install -r requirements-dev.txt
test: # test python code
	python setup.py develop && python -m pytest
rs: # compile rust code
	cargo