CUDA-HLL
=====

CUDA-HLL or CHLL is an simple proof of concept utility that constructs
HyperLogLog's for cardinality estimation using Nvidia CUDA for acceleration.
The motivation behind HyperLogLog's is they enable probabilistic cardinality
estimation in O(N) time with sub-linear memory usage.

This implementation makes use of the massively parallel nature of GPU's to
perform the hashing and parameter extraction of the HLL inputs in parallel.

Install
-------

Download and build from source:

    $ git clone https://armon@github.com/armon/cuda-hll.git
    $ cd cuda-hll
    $ make

At this point, `chll` will be an executable

Usage
-----

In the current implementation, CHLL expects to receive 36 byte wide inputs,
seperated by a newline (or any terminator). The 36 byte width corresponds with
a UUID. This can be easily tweaked in the constants, but it is a limitation of
the concept.

To generate some sample UUID's in Python, do:

    $ python
    > import uuid
    > fh = open("sample.txt", "w")
    > for x in xrange(100000):
    >   fh.write(str(uuid.uuid4()) + "\n")
    >
    > fh.close()
    > exit()


Then, once you are in the shell, you can invoke chll with the sample inputs:

    $ ./chll < sample.txt

You should be presented with an output like the following:

    Reading input...
    +7 msec: Copying to GPU...
    +43 msec: Hashing data... (100000 lines, 521 blocks, 192 threads)
    +50 msec: Extracting HLL values...
    +52 msec: Building HLL...
    +54 msec: Copying HLL...
    +54 msec: Estimating cardinality...
    Est: 102647.6 Raw: 102647.6
    +54 msec: Cleanup...
    +54 msec: Done

Performance
------------

Casual benchmarks were performed between 3 variations of this on a 2012 rMBP.
The first benchmark was the CHLL tool itself, the second was a non-CUDA
version (HLL), and the third was a simple awk script:

    awk '{ if (!($1 in s)) { s[$1]=1; c++; }} END { print "Count", c; }'

For each version, I tested against a file with 100K, 1MM, and 10MM rows.
The results are as follows:

* 100K rows
  * CHLL: 50ms
  * HLL:  25ms
  * AWK: 140ms

* 1MM rows
  * CHLL: 205ms
  * HLL:  215ms
  * AWK: 1480ms

* 10MM rows
  * CHLL: 1086ms
  * HLL:  2285ms
  * AWK: 16911ms



Troubleshooting
---------------

For CHLL to function the CUDA runtime environment
must be setup, and the computer must have a CUDA-enabled device. Otherwise,
and error like the following will occur::

    Reading input...
    +8 msec: Copying to GPU...
    +14 msec: Hashing data... (100000 lines, 521 blocks, 192 threads)
    Hashing failed: no CUDA-capable device is detected


References
-----------

* [Nvidia CUDA](http://www.nvidia.com/object/cuda_home_new.html)
* [HyperLogLog](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)

