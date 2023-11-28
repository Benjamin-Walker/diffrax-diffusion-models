# Score Based Diffusion Models

## Introduction

This repository implements score-based diffusion models, as described in [Score-Based Generative Modeling through 
Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), using equinox and diffrax. Currently, the
forward SDE is the variance preserving SDE (VP-SDE), 

$$ dX_t = -\frac{1}{2}\beta(t)X dt + \sqrt{\beta(t)}dW_t $$

where $W_t$ is a standard Brownian motion. The models are trained using de-noising score matching, 

$$ ||f_{\theta}(X_t, t) - \nabla_{X_t} \log p(X_t | X_0)||^2_2 $$

where $f_{\theta}$ is the neural network score function. In the case of the VP-SDE, p(X_t | X_0) is 
a Gaussian, $\mathcal{N}(\mu, \sigma)$. Therefore, 

$$ \nabla_{X_t} \log p(X_t | X_0) = \frac{\mu-X_t}{\sigma^2} $$

and the training objective can be rewritten as

$$ ||f_{\theta}(X_t, t) + \frac{Z}{\sigma(t)}||^2_2, $$

where $Z\sim \mathcal{N}(0, 1)$ and $\sigma(t) = \sqrt{1 - \exp(-\int_0^t\beta(s)\text{d}s)}$.


## Dependencies

- At least Python 3.9;
- [JAX](https://github.com/google/jax) for autodifferentiation;
- [Equinox](https://github.com/patrick-kidger/equinox) for neural networks, model building etc;
- [Diffrax](https://github.com/patrick-kidger/diffrax) for differential equations;
- [Optax](https://github.com/deepmind/optax) for optimisers;
- [Einops](https://github.com/arogozhnikov/einops/) for tensor rearrangement operations;