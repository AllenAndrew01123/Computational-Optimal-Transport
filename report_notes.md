## Part 1: LP OT

We solved the Kantorovich LP using three paradigms: HiGHS dual simplex, HiGHS interior point, and a manual NumPy PDHG implementation.

Small correctness checks were run for n = 3 and n = 4. In both cases, simplex and IPM returned identical objective values, zero marginal violation, nonnegative couplings, and total mass equal to 1.

For n = 50, 100, 200, 500, simplex and IPM again gave identical transport costs and zero marginal violation. PDHG produced costs extremely close to the LP optimum and marginal violations around 1e-7.

Timing observation: IPM scaled better than simplex on CPU. At n = 500, simplex took about 11.0 seconds, IPM took about 2.16 seconds, and manual PDHG took about 2.34 seconds.

## Part 2: Quadratic regularization

We solved the quadratically regularized OT problem using CVXPY with OSQP and CLARABEL. OSQP is ADMM-based, while CLARABEL is an interior-point conic solver.

For n = 200 and lambda = 0.1, both solvers returned nearly identical regularized objectives and transport costs. OSQP took about 39.1 seconds, while CLARABEL took about 0.24 seconds. Both produced very small marginal violations.

For the regularization path, lambda was varied from 10 to 1e-3. The unregularized transport cost <C, P_lambda> converged to the LP cost as lambda decreased. At lambda = 10, the cost was about 3.10e-3 above the LP cost. At lambda = 1e-3, the difference was about 4.91e-8.

### ADMM rho sensitivity

For fixed n = 200 and lambda = 0.1, we tested OSQP with rho values 0.01, 0.1, 1, and 10. The fastest was rho = 10, taking about 15.3 seconds, while rho = 1 took about 42.2 seconds. Larger rho improved runtime in this experiment but produced a slightly larger marginal violation, around 4.84e-7. All runs reached optimal status and nearly identical transport costs.


## Part 3: Entropic regularization

We solved entropic OT using Sinkhorn. A small test showed that POT Sinkhorn and a manual Sinkhorn implementation matched closely.

For the regularization path, naive Sinkhorn failed for small lambda because K = exp(-C/lambda) underflowed and the marginal violations became large. We therefore used POT's log-domain stabilized Sinkhorn.

With log-domain Sinkhorn, the transport cost converged toward the LP cost as lambda decreased from 10 to 0.01. At lambda = 10, the transport cost was about 2.96 above the LP cost. At lambda = 0.01, the gap was about 0.00314.

Runtime increased sharply for small lambda. For lambda = 10, the solve took about 0.005 seconds, while for lambda = 0.01 it took about 55 seconds. This shows the numerical difficulty of small entropic regularization.

### Sinkhorn convergence

For n = 200, manual Sinkhorn converged quickly for lambda = 1, taking 51 iterations and reaching marginal violation around 3.24e-11. For lambda = 0.1, it needed 451 iterations and reached violation around 7.80e-10. For lambda = 0.01, naive Sinkhorn failed: after 10000 iterations, the marginal violation was still about 0.279. This reflects the numerical instability of the Gibbs kernel K = exp(-C/lambda) for small lambda.

## Part 4: Gaussian OT

The closed-form Gaussian W2² for our chosen means and covariances is approximately 5.586.

We compared this value with discrete OT costs from empirical samples. For each n in {50, 100, 200, 500}, we repeated the experiment over 10 random seeds and plotted mean ± one standard deviation.

The empirical means were near the closed-form value but not monotone. This is expected because both the source and target point clouds are random finite samples. The standard deviation generally decreased as n increased.

We also plotted the n = 200 discrete OT arrows over the Gaussian covariance ellipses. The arrows show the empirical transport structure between sampled source and target points.