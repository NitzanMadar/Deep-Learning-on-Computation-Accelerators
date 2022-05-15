r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=32,
        gamma=0.98, 
        beta=0.5,
        learn_rate=0.003, 
        eps=1e-8,
        num_workers=0)
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=16,
              gamma=0.98,
              beta=0.5,
              delta=0.7,
              learn_rate=3e-3,
              eps=1e-8,
              num_workers=0) 
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

<font color='darkgreen'>


Subtracting a baseline $b$ in the policy-gradient helps to reduce variance because reducing the gradient by constant makes positive value that smaller than expected ($b$ functions as expected value) to be negative, where values that larger than the expected value will remain positive. In other words, it makes a sign-difference ($+$ and $-$) between negligible positive value and higher positive value which we aim to be at. This is effective if we want to enforce the model to learn trajectories that rewarded by values that larger than $b$

For example, if our model have some trajectories that lead to a reward of $~50$ and others to $~100$, we wish to take those trajectories that exceeded the $100$ and hence we can use $b$ that will help the model to do it. $b=90$ will change the value of $~50$ trajectories to be negative where the $~100$ will still positive and that will help us to choose this trajectories.

"""


part1_q2 = r"""
**Your answer:**

<font color='darkgreen'>

Recall:
$$ \pi(a|s)=\mathrm{Pr}(a_t=a|s_t=s) $$
$$ v_{\pi}(s) = \mathbb{E}[g_t(\tau))|s_t=s, \pi] $$
$$ q_{\pi}(s,a) = \mathbb{E}[g_t(\tau)|s_t=s, a_t = a, \pi] $$ 

Namely, 
* $v_{\pi}(s)$ averages the discounted reward per state $s$ according the policy 
* $q_{\pi}(s,a)$ averages the discounted reward per state $s$ and first action $a$ according the policy

Therefore the relationship between $v_{\pi}(s)$ and $q_{\pi}(s,a)$ can be presented as:  
$$
\begin{align}
v_{\pi}(s) &= \mathbb{E}[g_t(\tau)|s_t=s, \pi] \\ &= \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \mathbb{E}[g_t(\tau)|s_t=s, a_t = a, \pi] 
\\ &= \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \cdot q_{\pi}(s,a)
\end{align}
$$

In other words, $v$ is the average over the $q$s because $\pi(a|s)=\mathrm{Pr}(a_t=a|s_t=s)$.
So, we can use the action-values $q$ to approximate the state-value $v$ because it an average over the possible actions of the action-values for each action.

"""


part1_q3 = r"""
**Your answer:**

<font color='darkgreen'>

Recall:
   * Vanilla PG (`vpg`): No baseline, no entropy loss ($\hat\grad\mathcal{L}_{\text{PG}}(\vec{\theta})$)
   * Baseline PG (`bpg`): Baseline, no entropy loss ($\hat\grad\mathcal{L}_{\text{BPG}}(\vec{\theta})$)
   * Entropy PG (`epg`): No baseline, with entropy loss ($\hat\grad\mathcal{L}_{\mathcal{H}(\pi)}(\vec{\theta})$)
   * Combined PG (`cpg`): Baseline, with entropy loss ($\hat\grad\mathcal{L}_{\text{CPG}}(\vec{\theta})$)

1. First experiment results analysis of `vpg`, `epg`, `bpg` and `cpg`:
    1.1. **loss_p:** (policy-gradient as the negative average loss over trajectories)
    The Methods withous baseline considering (`vpg` and `epg`) starts with relatively low negative loss_p and improves the policy during the training and exceeded the zero values.
    Methods which uses baselint subtraction (`bpg` and `cpg`) shows small changess around $0$. It doesn't meants that the policy don't improve as shown it other graph, thats because the baseline subtraction.
    
    1.2. **loss_e:** (negative entropy loss)
    This graph is relevant only for the methods that uses entropy loss - `epg` which uses $\hat\grad\mathcal{L}_{\mathcal{H}(\pi)}(\vec{\theta})$ and `cpg` which uses $\hat\grad\mathcal{L}_{\text{CPG}}(\vec{\theta})$.
    The (negative) values of both methods rise as the networks process progress (get closer to zero).
    High entropy values (here, bigger in absolute value) means that the action probability distribution similar to uniform distribution, and reduction in entropy hint that the network converges to good policy and the network act more confidentlly.
    The subtraction of baseline $b$ in `cpg` speeds up the convergence rate, a good way to see it is the similar "undershoot" of both graphs, that happens earlier in `cpg` (around episode 500) compare to `epg` (around episode 1,000). Hence, `cpg` method is more efficient and has better values.
    
    1.3. **baseline:** (the baseline $b$)
    The $b$ values increases with every batch, it starts faster and converages slowly. Higher $b$ helps the network to increase the rewards. Here also, we can see that the method with baseline converges faster.
    
    1.4. **mean_reward:** 
    This graph shows an increase in mean reward for all methods.
    Thus, we can conclude that all the loss functions fit this problem (after choosing good hyperparameters).
    Comparting the method with and witout the entropy (`vpg` vs. `epg` and `bpg` vs. `cpg`), we can see that at the entropy helps the process without baseline, but with the baseline the graphs look pretty similar.
    Additionally, as mentioned before, using baseline helps the training to converge faster (comparing `vpg` vs. `bpg` and `epg` vs. `cpg`) - the mean reward is much higher at the earlier episodes but in higher episodes the the curves reach pretty similar values.


2. Comparison between regular policy-gradient `cpg` and advantage actor-critic `AAC`:

    2.1. **loss_p:**
    `AAC` starts with lower values (similar to methods withous baseline) but quickly improve and reches higher values than `cpg`, which means a lower trajectory loss. We can conclude that `AAC` is better model and approximate better the state value (better baseline).
    
    2.2. **loss_e:**
    The `AAC` method here has more fluctuating curve, which probably depend on hyperparameters. But, we can see here that the `AAC` entropy loss reaches smaller absolute entropy loss, and hence has better results compare to `cpg`. The difference strongly depend on the baseline choose, because this is the only different between $\hat\grad\mathcal{L}_{\text{CPG}}(\vec{\theta})$ and $\hat\grad\mathcal{L}_{\text{AAC}}(\vec{\theta})$.
    
    2.3. **baseline:** not relevant for `AAC`, same as 1.3.
    
    2.4. **mean_reward:**
    shows similar performance for `AAC` and `cpg` at the earlier episodes but later the `AAC` shows a decrease (which looks like at the same point of the instability at the entropy loss), but it look like it fix itself at the end. We can say that this task is probably simple enought that the `cpg` get rewards which are not worse than the `AAC`.

"""
