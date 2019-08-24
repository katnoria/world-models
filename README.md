# world-models
Implementation of worldmodels paper by Ha and Schmidhuber

# Ingredients

V: Autoencoder to compress the environment state Z (from high dimension to lower dimension)
M: MDN-RNN to predict the next compressed state Z based on current state Z and action A
C: Simple single layer network to predict the next action

# Installation

We use Ray to achieve parallelism.
>Ray is a fast and simple framework for building and running distributed applications.

Ray internally uses plasma object store to allow memory sharing and avoid object copy.

By default, plasma store starts with 5GB of Memory, if you're short on memory then you can set the size using following command.

```
plasma_store -m 1000000000 -s /tmp/plasma
```

>The -m flag specifies the size of the store in bytes, and the -s flag specifies Zthe socket that the store will listen at. Thus, the above command allows the Plasma store to use up to 1GB of memory, and sets the socket to /tmp/plasma.

source: https://arrow.apache.org/docs/python/plasma.html

Set plasma store manually
```
conda install -c conda-forge pyarrow
```