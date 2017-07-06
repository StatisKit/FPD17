FPD17: Supplementary material
#############################

This repository contains **Jupyter** `notebooks <https://jupyter.readthedocs.io/en/latest/index.html>`_ for reproducing simulations and analyzes performed in the article *Multinomial Splitting Models for Multivariate Counts* written by:

* Pierre Fernique,
* Jean Peyhardi,
* Jean-Baptiste Durand.

Test it !
=========

Using **Docker** `images <http://docs.mybinder.org/>`_. or a **Binder** `server <https://docs.docker.com/>`_, we are able to provide ways to reproduce the article studies without installing the **StatisKit** software suite.
    
Online with **Binder**
----------------------

To reproduce the studies online, follow this `link <http://mybinder.org/repo/statiskit/fpd17>`_.
Note that the **Binder** server might be outdated but can be updated by following this `link <http://mybinder.org/status/statiskit/fpd17>`_ and clicking on :code:`rebuild`.

On your computer with **Docker**
--------------------------------

To reproduce the studies with **Docker** use this `image <https://hub.docker.com/r/statiskit/FPF17/tags>`_.
After `installing <https://docs.docker.com/engine/installation/>`_ **Docker**, you can type the following commands in a shell:

.. code-block:: console

    docker run -i -t -p 8888:8888 statiskit/FPD17:latest
    jupyter notebook index.ipynb --ip='*' --port=8888 --no-browser
   
You can then view the **Jupyter** notebooks by following this `link <http://localhost:8888/notebooks/index.ipynb>`_, or http://<DOCKER-MACHINE-IP>:8888/notebooks/index.ipynb if you are using a **Docker** Machine VM (see this `documentation <https://docs.docker.com/machine/>`_ for more informations).

Install it !
============

You can also install on your computer all required packages to reproduce these studies.
In order to ease the installation of these packages on multiple operating systems, the **Conda** `package and environment management system <> `_ is used.
For more information refers to the **StatisKit** software suite documentation concerning prerequisites to the `installation <http://statiskit.readthedocs.io/en/latest/user/install_it.html>`_.
Then, to install the required packages, proceed as as follows:

1. Clone this repository,

   .. code:: console
   
     git clone https://github.com/StatisKit/FPD17
     
2. Enter in the cloned repository,

   .. code:: console
   
     cd FPD17
     
3. Install the **Conda** environment,

   .. code:: console

     conda env create -f environment.yml
  
4. Activate the `FPD17` environment as precised in your terminal.

5. Launch the **Jupyter** index notebook,

   .. code:: console

     jupyter notebook index.ipynb
     
6. Execute this `index.ipynb` notebook to execute all examples or navigate among referenced notebooks to execute them separatly.
