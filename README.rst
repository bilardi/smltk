Getting started
===============

smltk (Simple Machine Learning Tool Kit) package is implemented for helping your work during

* data preparation
* testing your model

The goal is to implement this package for each step of machine learning process that can simplify your code.

It is part of the `educational repositories <https://github.com/pandle/materials>`_ to learn how to write stardard code and common uses of the TDD.

Installation
############

If you want to use this package into your code, you can install by python3-pip:

.. code-block:: bash

    pip3 install smltk
    python3
    >>> from smltk.metrics import Metrics
    >>> help(Metrics)

The package is not self-consistent. So if you want to contribute, you have to download the package by github and to install the requirements

.. code-block:: bash

    git clone https://github.com/bilardi/smltk
    cd smltk/
    pip3 install --upgrade -r requirements.txt

Read the documentation on `readthedocs <https://smltk.readthedocs.io/en/latest/>`_ for

* API
* Usage
* Development

Change Log
##########

See `CHANGELOG.md <https://github.com/bilardi/smltk/blob/master/CHANGELOG.md>`_ for details.

License
#######

This package is released under the MIT license.  See `LICENSE <https://github.com/bilardi/smltk/blob/master/LICENSE>`_ for details.
