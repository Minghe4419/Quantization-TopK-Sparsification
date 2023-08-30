# Bias Variance Tradeoff Gradient Compression


## Description
In our project, we investigate in the tradeoff between variance and biasedness of quantization in FL. We focus on the topK sparsification quantizors and iid data that are uniformly and normally distributed. In the code, we observe the behaviors of weight of the first quantizer $w1$ and Mean Square Error of the quantization with the change of K value we choose for the first quantizes $k1$. 

**Quadratic Programming Problem**: 

- In order to find the optimal weight with the minimum MSE, we use quadratic programming solver. So we need expectation of X and $X^2$ with the condition that X is among the K highest values in each entry. In this case, we can obtain both quadratic part $P$ and linear part $q$ of the function. 
- The expectations can be found using both Monte Carlo Simulation and Analytic Expression in the uniformly distributed data. For normally distributed ones, we can only use Monte Carlo Simulation because there is no explicit analytic expression for it. Besides, we give equality constraint to our quadratic programming problem to guarantee the sum of quantizers is unbiased $E(Q(v)) = E(v)$ and the sum of weight is number of quantizers. 

- Asymptotic Case: We investigate in the asymptotic case by letting the number of values in each entry(d) to approach infinity and sum of means to be 0. With a infinite $d$, $P$ becomes a diagonal matrix with k value times conditional expectation of $X^2$. Meanwhile, $q$ becomes zero with an inifinite $d$.

## Functions
The file ''Behavior_of_Asymptotic_case_in_QPF_using_MCS /w1_MSE_for_uniform_normalDist__includingAsymptotic___1_.ipynb'' investigate in the Bahvior of Asymptotic case in our Quadratic Programming Functions using Monte Carlo Simulation. It focuses on the uniformlly and normally distributed data. By simulating their asymptotic cases, we are able to observe how $w_1$ and $MSE$ behaves as the $k_1$ becomes large/dominant among the $k$s for each entry.
- `DataEntryCreator`: This function is able to randomly create data entry with iid values in uniform, normal and exponential distribution with specify their parameters.
### Monte Carlo Simulation
- `isFirstEntryAmongTopK_mce`: This function can help us to know which iteration finds the value that is among K highest values.
- `expectation_of_xAndx2`: This function can simulate the conditional expectation of X and $X^2$ by taking the average of X and $X^2$ when X is among K highest values. \
_Limit_: since we need to iterate the simulation $2^9$ times for each data entry(specify with parameters of distribution), the simulation will be very costly. In this case, I use np.partition in isFirstEntryAmongTopK_mce with a cost of O(n). However, there might be other method to further improve on the cost which allows us to improve the precision with larger datasets and more iterations.

### Quadratic Programming
We investigate the solution to quadratic programming problem with equality constraint, without equality constraint, and with asymptotic case. 
- `eQTQ_ks`, `eZTQ_ks`, `equal_constr_ks` are functions returning quadratic part$P$, linear part$q$, equality constraint$A, b$ of our qudratic programming function.
- `quad_prog_func`: We can determine either create normal or uniform data by controlling the function's parameter. This function can return the optimal weight, corresponding MSE, conditional expectations of $x$ and $x^2$ and mean of $x$.
- `lagra_approxm_qp`: Similar with `quad_prog_func`, while only returning a list of first weights`W` in asymptotic case.
- `quad_prog_func_uncstr`: Similar with `quad_prog_func`, while no longer having equal constraint.
- `quad_prog_func_approxAsym`: the Asymptotic case is approximate by approaching d to positive inifinity. In this case, we modifies the equality constraint for the first data entry by using `A_matrix[0][0] = 0`. 
For the quadratic programming in all cases, I create both analytic method version(Uniform Data Only) - end with `_a` in the function name - and monte carlo simulation version(Both Uniform and Normal Data).

#### Plot of $w_1$ vs $k_1$ and $MSE$ vs $k_1$ in different cases
- `K_3plots`: This function will plot $w_1$ vs $k_1$ and `MSE` vs $k_1$ in cases:
    - First weight $w_1$ of quadratic programming function with constraint vs $k_1$
    - Result `MSE` of quadratic programming function with constraint vs $k_1$
    - Result `MSE` of quadratic programming function without constraint vs $k_1$
    - Difference between the Result of quadratic programming with and without constraint vs $k_1$
    - First weight $w_1$ of quadratic programming function with d goes to infinity vs $k_1$
    - Result `MSE` of quadratic programming function with d goes to infinity vs $k_1$

In conclusion, we can tell that behavior of $w_1$ vs $k_1$ in asymptotic case is similar with the one of quadratic programming function with constraint. Behavior of `MSE` vs $k_1$ in asymptotic case is similar with the one of quadratic programming function without constraint.
- `K_3plots_a` is the function using analytic method version(Uniform Data Only) of quadratic programming functions.

### Limits
- Why behavior of `MSE` vs $k_1$ in asymptotic case is different from the one of quadratic programming function with constraint remains unexplained or uninterpreted.
- When simulating the data entries, we only observed normally and uniformlly distributed entry. And $w_1$ and `MSE` only performs with a clearer trend when the data entries' distributions are approaching standard uniform and standard normal distribution.
