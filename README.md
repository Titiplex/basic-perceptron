# TP1 Supervised Learning

You can find this repository on [GitHub](https://github.com/Titiplex/basic-perceptron)

## Requirements

You may have to download [gnuplot](https://sourceforge.net/projects/gnuplot/) and add it to your path environment
variables.

This project uses C++ 23 and Cmake.
Cmake will automatically download and install [Matplot++](https://alandefreitas.github.io/matplotplusplus/), which is
used for heatmap generation.

Alternatively, you may use ``make`` to configure, build and run the project.

## Configure

Cmake:

````shell
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
````

Makefile:

````shell
make configure
````

## Build

Cmake:

````shell
cmake --build build-ninja --parallel
````

Makefile:

````shell
make build
````

## Execute

In your shell:

````shell
./build-ninja/TP1
````

Or if you want to use ``make`` in your shell:

````shell
make run
````