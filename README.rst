.. image:: https://img.shields.io/badge/License-Apache%202.0-yellow.svg
   :target: https://opensource.org/licenses/Apache-2.0
   
.. image:: https://travis-ci.org/StatisKit/FPD18.svg?branch=master
   :target: https://travis-ci.org/StatisKit/FPD18
  
.. image:: https://ci.appveyor.com/api/projects/status/bwc7elajp21arif0/branch/master
   :target: https://ci.appveyor.com/project/pfernique/fpd18/branch/master

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/StatisKit/FPD18/master?filepath=share%2Fjupyter%2Findex.ipynb

.. warning:: 

   Currently down, will be restored the 2018/02/11
   
Material for the paper entitled "Splitting models for multivariate count data"
==============================================================================

This repository contains supplementary material for the reproducibility of computational studies performed in the article "Splitting distributions for multivariate count data" written by:

* Pierre Fernique,
* Jean Peyhardi,
* Jean-Baptiste Durand.

This article has been pre-published in the `arXiv <https://arxiv.org/abs/1802.02074>`_ e-print service.
Here is the the citation formated as the bibtex standard.

.. code-block:: bibtex

   @ARTICLE{2018arXiv180202074F,
            author = {{Fernique}, P. and {Peyhardi}, J. and {Durand}, J.-B.},
            title = "{Splitting models for multivariate count data}",
            journal = {ArXiv e-prints},
            archivePrefix = "arXiv",
            eprint = {1802.02074},
            primaryClass = "math.ST",
            keywords = {Mathematics - Statistics Theory, Mathematics - Probability},
            year = 2018,
            month = feb,
            adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180202074F},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

These studies are formatted as pre-executed **Jupyter** `notebooks <https://jupyter.readthedocs.io/en/latest/index.html>`_.
Refers to the `index.ipynb <share/jupyter/index.ipynb>`_ notebook which presents and references each study.

Test it !
=========

Using **Docker** `images <https://docs.docker.com/>`_  and **Binder**  `servers <http://docs.mybinder.org/>`_ , we are able to provide ways to reproduce the article studies without installing the **StatisKit** software suite.
    
Online with **Binder**
----------------------

.. warning::

   Currently, **Binder** does not work with *R* kernel
   
To reproduce the studies online, use this `server <https://mybinder.org/v2/gh/StatisKit/FPD18/master?filepath=share%2Fjupyter%2Findex.ipynb>`_.

On your computer with **Docker**
--------------------------------

To reproduce the studies with **Docker** use these `images <https://hub.docker.com/r/statiskit/fpd18/tags>`_.
After `installing <https://docs.docker.com/engine/installation/>`_ **Docker**, you can type the following commands in a shell:
  
* For the *Python 2* version 

  .. code-block:: console

    docker run -i -t -p 8888:8888 statiskit/fpd18:latest-py2k
   
* For the *Python 3* version 

  .. code-block:: console

    docker run -i -t -p 8888:8888 statiskit/fpd18:latest-py3k
  
Then, follow the given instructions.
    
Install it !
============
  
You can also install required packages on your computer to reproduce these studies.

.. warning::

  To use the examples without **Docker** you must first follow instructions available on this `page <http://statiskit.readthedocs.io/en/latest/developer/configure.html>`_.
  
In order to ease the installation of these packages on multiple operating systems, the **Conda** `package and environment management system <https://conda.io/docs/>`_ is used.
For more information refers to the **StatisKit** software suite documentation concerning prerequisites to the `installation <http://statiskit.readthedocs.io/en/latest/user/install_it.html>`_ step.
Then, to install the required packages, proceed as as follows:

1. Clone this repository,

   .. code:: console
   
     git clone --recursive https://github.com/StatisKit/FPD18
     
2. Create a **Conda** environment containing the meta-package :code:`fpd18`,
      
   .. code:: console

       conda create -n fpd18 fpd18 python=2 -c statiskit/label/unstable -c r -c defaults --override-channels
     
3. Activate the **Conda** environment as advised in your terminal.

4. Enter the directory containing **Jupyter** notebooks,

   .. code:: console
   
     cd FPD18
     cd share
     cd jupyter
     
5. Launch the **Jupyter** the `index.ipynb <jupyter/index.ipynb>`_ notebook,

   .. code:: console

     jupyter notebook index.ipynb
     
6. Execute the `index.ipynb <share/jupyter/index.ipynb>`_ notebook to execute all examples or navigate among referenced notebooks to execute them separately.
