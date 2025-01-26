# Monte-Carlo Tree Search
Each edge $(s, a)$ in the search tree stores three values:
1. The visit count $N(s, a)$
2. The total action-value $W(s, a)$
3. The prior probability $P(s, a)$

A single "simulation" consists of three phases:

## Select
Beginning from the root node $s_0$ we traverse down the tree until we find a leaf node 
$s_L$. At each step we select the edge $(s_t, a)$ which maximises $Q(s_t, a) + \alpha U
(s_t, a)$ where $Q(s, a) = \frac{W(s, a)}{N(s, a)}$ is the mean action-value and $U(s, 
a) = P(s, a)\frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$ is the PUCT bonus, and 
$\alpha > 0$ is a hyperparameter.

## Expand and Evaluate
The leaf node $s_L$ is added to a queue for neural network evaluation. Positions in 
the queue are evaluated by the neural network in batches. The simulation thread is locked 
until evaluation completes. The leaf node is expanded and each edge $(s_L, a)$ is 
initialised to:
1. $N(s_L, a) = 0$
2. $W(s_L, a) = 0$
3. $P(s_L, a) = p_\gamma(a | s_L)$

The neural network's predicted value $v_\theta(s_L)$ is then backed up:

## Backup
We move back up the tree until we reach $s_0$, updating like so:
1. $N(s_t, a) \leftarrow N(s_t, a) + 1$
2. $W(s_t, a) \leftarrow W(s_t, a) + v_\theta(s_L)$


## Play
After $n$ simulations have completed, a move is selected based on the probability 
distribution $\pi(a | s_0) = \frac{N(s_0, a)^{1 / \tau}}{\sum_b N(s_0, b)^{1 / \tau}}$ 
where the temperature $\tau > 0$ is a hyperparameter.

## Loss
Given a game $s_0, ..., s_T$ with move probabilities $\pi_0, ..., \pi_T$ and outcome 
$z$, the loss is:
$$
\mathcal{L}(s_t, \pi_t, z) = -[z\log(v_\theta(s_t)) + (1 - z)\log(v_\theta(s_t)) + 
\pi_t^\intercal \log(p_\gamma(s_t))]
$$