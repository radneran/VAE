# Variational Autoencoder
Based on variational inference

$$P(Z|X) = \frac{P(Z,X)}{P(X)}$$

## Some basic information theory
### Information 
$I = -log(P(x))$
* Measures the factor of uncertainty removed when x is known
* 1 bit can be thought of as information that reduces uncertainty
    by a factor of 2
E.g. Let's say there is 50% chance of weather being sunny and 
50% chance of rain tomorrow. When the weather station tells us it is going to be sunny, they have given us 1 bit of information.

> Uncertainty reduction is the inverse of the event's probability

E.g. If the weather probabilities are sunny 75% and rain 25%,
finding out that it is going to be rainy will reduce our 
uncertainty by $1/0.25 = 4$. This corresponds to $log_2(4) = 2$ bits of information $log_2(\frac{1}{0.25}) = -log_2(0.25)$

### Entropy 
$H = \sum_x-log(P(x)) * P(x))$  
* Can be thought of as average amount of information relayed by a certain distribution
E.g in the above case, the weather station on average transmits
$$ H = 0.75 \times -log(0.75) + 0.25 \times (-log(0.25)) 
  = 0.81$$ bits of useful information

### Cross-entropy
$$H(P,Q) = \sum_x P(x)(-log(Q(x))$$

E.g Lets say we use 2 bits to encode our weather prediction
this can be thought of as us predicting the weather to have
a 25% chance of either being sunny or rainy
The average number of actual bits sent is
$H = 0.75 \times 2 + 0.25 \times 2 = 2 $bits. If using different number of bits for the different predictions $H = 0.75 \times 2 + 0.25 \times 3 = 2.25 $bits 

Based on the entropy and cross-entropy, we can see that our _predicted_ probability distribution Q(x) differs from the _actual_ probability distribution P(x) by 
$KL(P||Q) = 2.25 - 0.81 = 1.54 $bits  
If predictions are perfect i.e. Q(x) = P(x), then H(P,Q) = H(P)  
Therefore, $H(P,Q) = H(P) + KL(P||Q)$  
$KL(P||Q)$ means KL-divergence of Q w.r.t P

\begin{align}
KL(P||Q) &= H(P,Q) - H(P)\\
         &= \sum_x P(x)(-log(Q(x)) -  \sum_xP(x)(-log(P(x))\\
         &= \sum_{x} P(x)(-log(Q(x) - (-log(P(x)))\\
         &= \sum_x P(x)(log(P(x)) - log(Q(x)))\\
         &= \sum_x P(x)(log(\frac{P(x)}{Q(x)}))\\
\end{align}
#### Some properties of KL-divergence
1. $KL(P||Q)$ is alwaysgreater than or equal to 0
2. $KL(P||Q)$ is not the same as $KL(Q||P)$


## Variational Bayes

$$P(Z|X) = \frac{P(Z,X)}{P(X)} = \frac{P(X|Z)P(Z)}{P(X)}$$

We don't know P(X). If we were to compute it,
$P(X) = \int{P(X|Z)P(Z)dZ}$
* Intractable in many cases
* If distributions are high dimensional, integral is multi-integral

Thus, we can try to approximate the distribution. One method to approximate is Monte Carlo method (Gibbs sampling and other sampling methods) which is unbiased with high variance.

Another is variational inference which has low variance but is biased

1. Approximate P(Z|X) with Q(Z) that is tractable e.g. Gaussian
2. Play with the parameters of Q(Z) in a way that it gets close enough to P(Z|X)

This brings us to the following objective of minimizing 


\begin{align}
KL(Q(Z)||P(Z|X)) &= \sum_z Q(Z)log(\frac{Q(Z)}{P(Z|X)})\\
                 &= - \sum_zQ(Z) log(\frac{P(Z|X)}{Q(Z)})\\
                 &= - \sum_z Q(Z) log(\frac{P(X,Z)}{P(X) Q(Z)})\\
                 &= - \sum_z Q(Z) (log(\frac{P(X,Z)}{Q(Z)}) - log(P(X)))\\
                 &= - \sum_z Q(Z) log(\frac{P(X,Z)}{Q(Z)}) + log(P(X))\\
\end{align}
\begin{align}
\therefore log(P(X))    &= KL(Q(Z)||P(Z|X)) + \sum_z Q(Z)log(\frac{P(X,Z)}{Q(Z)})\\
                 &= KL(Q(Z)||P(Z|X)) + L\\
\end{align}


As $log(P(X))$ is a constant, to minimize $KL(Q(Z)||P(Z|X))$,
we just need to maximize $L$.

$$\because KL(Q(Z)||P(Z|X)) \geq 0$$,  
$$L \leq P(X)$$ Thus, L is a lower bound of P(X).

\begin{align}
L &= \sum_z Q(Z) log(\frac{P(X,Z)}{Q(Z)})\\
  &= \sum_z Q(Z) log(\frac{P(X|Z)P(Z)}{Q(Z)})\\
  &= \sum_z Q(Z)(log(P(X|Z)) + log(\frac{P(Z)}{Q(Z)}))\\
  &= \sum_z Q(Z) log(P(X|Z))) + \sum_z Q(Z) log(\frac{P(Z)}{Q(Z)})\\
\end{align}

$$\sum_z Q(Z) log(P(X|Z))) = E_{Q(Z)}P(X|Z)$$
$$\sum_z Q(Z) log(\frac{P(Z)}{Q(Z)}) = -KL(Q(Z)||P(Z))$$
Representing L as an autoencoder

X --> Q(Z|X) --> Z --> P(X|Z) --> X'

$E_{Q(Z)}P(X|Z)$ term acts as reconstruction error.
$P(X|Z)$ is deterministic meaning one input will get the same output all the time. Thus, it can be considered $P(X|X')$.

If $P(X|X')$ is gaussian
$$P(X|X') = e^{-|X - X'|^2}$$
$$log(P(X|X')) = -|X - X'|^2$$ --> L2 loss

If Bernoulli distribution, will be similar to cross-entropy

So far the network is all deterministic
> To make it probabilistic, 
    encoder should not parametrize Z but instead the parametrize
    the distribution that generates Z i.e. $\mu$ and $\sigma$


