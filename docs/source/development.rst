Development
===========

Your contribution is important but you have to follow same steps.

It will be approved changes or additions with

* methods containing title, arguments and returns descriptions
* the relative unit tests

Run tests
#########

It is important to test your code before to create a pull request.

.. code-block:: bash

    cd smltk/
    pip3 install --upgrade -r requirements.txt
    python3 -m unittest discover -v
