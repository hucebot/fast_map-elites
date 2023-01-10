# fast_map-elites
The fastest MAP-Elites implementation?

On a MacBook 2020, about`0.5 s` for 1M evaluations of the kinematic arm.

## Why doing this?
- benchmarking new ideas in a few minutes of computation
- hyperparameter tuning
- online optimization

And it is good to have implementations that are not wasteful when possible!


## Algorithms

- MAP-Elites: Mouret JB, Clune J. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
- Using centroids: Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.
- Variation operator: Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation. InProceedings of the Genetic and Evolutionary Computation Conference 2018 Jul 2 (pp. 149-156).
- Bounce back operator: Nordmoen J, Nygaard TF, Samuelsen E, Glette K. On restricting real-valued genotypes in evolutionary algorithms. InInternational Conference on the Applications of Evolutionary Computation (Part of EvoStar) 2021 Apr 7 (pp. 3-16). Springer, Cham.

## Compilation:
- `clang++  -DUSE_BOOST -DUSE_TBB  -ffast-math  -O3 -march=native -std=c++14 -I /usr/local/include/eigen3 map_elites.cpp -o map_elites -ltbb`
- `-DUSE_BOOST`: use Sobol centroids instead of random centroids (this requires boost installed)
- `-DUSE_TBB`: use TBB for multithreading (this is slightly slower in many cases!)