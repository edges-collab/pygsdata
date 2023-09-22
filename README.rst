========
read_acq
========
**Read EDGES ACQ spectrum files.**

.. image:: https://travis-ci.org/edges-collab/read_acq.svg?branch=master
    :target: https://travis-ci.org/edges-collab/read_acq
.. image:: https://codecov.io/gh/edges-collab/read_acq/branch/master/graph/badge.svg
    :target: https://travis-ci.org/edges-collabcodecov.io/gh/edges-collab/read_acq
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Installation
============

In a new/existing python environment, run
``pip install git+https://github.com/edges-collab/read_acq``.

If you wish to develop ``read_acq``, do the following::

    git clone https://github.com/edges-collab/read_acq
    cd read_acq
    pip install -e .

Usage
=====

``read_acq`` can be used in a Python interpreter or directly via the command line.

CLI
---

To use the CLI, a single command is provided:

    acq convert <data.acq> [<data2.acq> ...] [-f format1 [-f format2]]

This will convert the file(s) provided to all formats provided, and place any resulting
files in the same location as the original datafile(s) (but with different extension).
The default format is 'mat'.
The command can be run from anywhere on the system, and the file given can be a
relative or absolute path.

Multiple data files can be given, and each will be converted. Wildcards may also be
used in any of the filenames, eg.::

    acq convert data/*.acq

Library
-------

The main point of entry for the Python interface is ``decode_file``::

    >>> from read_acq import decode_file
    >>> data = decode_file("my_data.acq", write_formats=[])

By default, this function will also write the file in ``.mat`` format, but you can turn
that off by providing ``write_formats=[]``. The output is a ``numpy`` array of the data.
Several more options are provided, use ``help(decode_file)`` in an interpreter to see
all the options.
