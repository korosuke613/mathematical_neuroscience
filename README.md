﻿# mathematical_neuroscience
Mathematical Neuroscience Course Materials ([日本語](READMEj.md))

    git clone "https://github.com/date333cs/mathematical_neuroscience"


### Associative Memory

- [projects](./docs/project160415associative.pdf)

- Purpose: reproduce Fig.1 and Fig.2 of Amari & Maginu (1988).

- an example of memory allocation:  [mn_test.c](mn_test.c)
- templates: [mn_asm000.c](mn_asm000.c), [mn_asm001.c](mn_asm002.c), [mn_asm002.c](mn_asm002.c)
- for gnuplot [gp001](gp001)
- data structure (n neurons, m memory patterns, i,j=0,...,n-1, a=0,...,m-1)
 - int x[i] :  current state
 - int ux[i] : next state 
 - double w[i][j] : connection weights
 - int p[a][i] : memory items

### Self-organizaing MAP


<img src="https://github.com/date333cs/mathematical_neuroscience/blob/master/result_som_102.png" height="220px" align="left">

- [projects](./docs/project160513neural_som.pdf)

- Input: 2D uniform distribution, Neural Field (Output): 1D (n=100)

- templates: [mn_som000.c](mn_som000.c), [mn_som001.c](mn_som001.c),
- code [mn_som101.c](mn_som101.c) (written in the class room at 2016.5.20)
- code [mn_som102.c](mn_som102.c), [gp_som102](gp_som102)

- for MNIST [templates](http://www.cs.miyazaki-u.ac.jp/~date/lectures/2015neural/mnist/index.html)


### Boltzmann Machine

- [projects](./docs/project160610bolzmann.pdf)


- templates: [mn_bm001.c](mn_bm001.c), [mn_bm002.c](mn_bm002.c), [mn_bm003.c](mn_bm003.c), 

- data structure (n neurons)
 - int x[i] :  current state
 - double w[i][j] : connection weights
 - int s[a]: (a counter) for computing how many times each state appear 


### Stochastic gradient of piecewise linear discriminant functions, Amari (1967) 

- Purpose: reproduce Fig.5-1 (p.120) of Amari (1967).

+



