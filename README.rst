FPD17: Material for reproducibility 
===================================

This repository contains examples, data, and a **Binder**-compatible environment specification file for reproducing simulations and analyses performed in the article *Multinomial Splitting Models for Multivariate Counts* written by P. Fernique, J. Peyhardi and J.-B. Durand.

Test it !
---------

To :

* Play with these examples, follow this `link <http://mybinder.org/repo/statiskit/fpd17>`_.
* Update the **Binder** server, follow this `link <http://mybinder.org/status/statiskit/fpd17>`_ and click on :code:`rebuild`.

Install it !
------------

You can also install locally all required packages to run these examples as follows:

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
     
6. Execute this notebook to execute all examples or navigate among linked notebooks to execute them separatly.
