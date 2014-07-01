
* IN PROGRESS...

* MNIST demo using baf

    # two args: datadir outfile
    python mnist2baf.py . mnist.baf

Now, 

    baf info mnist.baf


* Testing on many languages

Now that we have the data and the model in baf files,
we can run tests in many langauges very easily.

All of these should give,

    Error: 7.76% (776 / 10000)

** Python

    cd python
    python runtest.py

** C

    cd c
    gcc -o tmp runtest.c libbaf.o  &&  ./tmp

