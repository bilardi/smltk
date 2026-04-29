# Simple Machine Learning Tool Kit Makefile

PACKAGE_NAME = "smltk"
YOUR_USERNAME = "bilardi"

.PHONY: help # print this help list
help:
	grep PHONY Makefile | sed 's/.PHONY: /make /' | grep -v grep

.PHONY: unittest # run unit tests
unittest:
	python3 -m unittest discover -v

.PHONY: clean # remove packaging files
clean:
	rm -rf build dist *.egg-info; rm -rf */*pyc; rm -rf */*/*pyc; rm -rf */__pycache__

.PHONY: doc # build documentation
doc: 
	cd docs; make html; cd -

.PHONY: localbuild # build package locally
localbuild: clean
	pip install black build
	black .
	python3 -m build; pip install dist/*whl

.PHONY: buildtest # build package on testpypi
buildtest: clean
	pip install --upgrade black build twine
	black .
	python3 -m build; twine upload --repository testpypi dist/*

.PHONY: installtest # install package from testpypi
installtest:
	mkdir -p test; cd test; python3 -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps $(PACKAGE_NAME); cd -

.PHONY: build # build package on pypi
build: clean
	pip install --upgrade black build twine
	black .
	python3 -m build; twine upload dist/*

.PHONY: install # install package from pypi
install:
	python3 -m pip install --upgrade $(PACKAGE_NAME)
