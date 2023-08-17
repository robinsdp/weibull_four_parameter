# weibull_four_parameter
Calculates the Maximum Likelihood regression coefficients for a custom 4 parameter Weibull Distribution. The basis for this distribution and derivation may be found below from the paper below:

S. Tous, E. Y. Wu and J. Sune, "A Compact Model for Oxide Breakdown Failure Distribution in Ultrathin Oxides Showing Progressive Breakdown," in IEEE Electron Device Letters, vol. 29, no. 8, pp. 949-951, Aug. 2008, doi: 10.1109/LED.2008.2001178.

Link to one online source: https://ieeexplore.ieee.org/document/4571158

Below is a parameterization of this distribution from the paper. Please note that the methodology for finding the maximum likelihood uses Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm because the search space for finding the parameters is unclear (I have not checked to see if there is a better search strategy to converge quicker) as opposed to standard least squares regression, especially within the exponential family of distributions, in which other more efficient strategies may be used.

![unnamed](https://github.com/robinsdp/weibull_four_parameter/assets/2322478/fb3a44da-6b48-4df1-9d4e-1ffd45218509)

