.. _getting_started:

===============
Getting started
===============

The python package to solve scheduling problems of all categories (Single machine, Parallel machines, Flowshop and Jobshop) under different constraints combination.

Python version support
======================

The use of **pyscheduling** requires a minimum python version of 3.9 to allow the type hinting.


Installation
============

.. panels::
    :card: + install-card
    :column: col-12 p-3

    Installing stable release (v0.1.3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    pyscheduling can be installed via pip from `PyPI <https://pypi.org/project/pyscheduling>`__.

    ++++++++++++++++++++++

    .. code-block:: bash

        pip install pyscheduling

Basic example
============

.. panels::
    :column: col-12 p-3

    The following is a basic code example of how to use **pyscheduling** where we create a single machine problem consisting of minimizing the total weighted lateness.

    ++++++++++++++++++++++

    .. code-block:: bash

        import pyscheduling.SMSP.interface as sm

        problem = sm.Problem()
        problem.add_constraints([sm.Constraints.W,sm.Constraints.D])
        problem.set_objective(sm.Objective.wiTi)
        problem.generate_random(jobs_number=20,Wmax=10)
        solution = problem.solve(problem.heuristics["ACT"])
        print(solution)

    ---
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    How to use pyscheduling?
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    See examples of use here :

    .. link-button:: samples.html
        :type: url
        :text: Examples
        :classes: btn-secondary stretched-link

    ---
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    API reference
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Full code documentation here :

    .. link-button:: autoapi/index.html
        :type: url
        :text: API reference
        :classes: btn-secondary stretched-link