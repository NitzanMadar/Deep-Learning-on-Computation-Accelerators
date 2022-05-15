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
By subtracting a baseline that is independent of the policy parameters, we reduce the overall gradients while maintaining an unbiased gradient estimate.
This in turn reduces the variance.

The baseline $b$ serves as the expected return value, allowing the model to treat insignificant positive rewards as _negative_ return values.
This is particularely effective if we wish to enforce that the model learns trajectories that lead to rewards greater than $b$.  
Example: A batch containing several trajectories resulting in a reward of $1,2$ and a few others resulting in rewards exceeding $50$. If we negate the small rewards, we can ensure that the model gravitates towards the higher rewards.


"""


part1_q2 = r"""
**Your answer:**

<font color='darkgreen'>
Since $v_{\pi}(s)$ averages over the first action according to the policy and $q_{\pi}(s,a)$ fixes the first action and continues as prescribed by the policy - we can use the q-values (action-values) to approximate the state-values as an average over the possible actions of the action-values produced by each action:

$$ v_{\pi}(s) = \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \cdot q_{\pi}(s,a) $$
where the policy $\pi(a|s)$ is the probability of taking action $a$ at state $s$.
To show that the approximation is valid:  

Recall the definitions of the value function $v_{\pi}(s)$ and action value function $q_{\pi}(s,a)$ under policy $\pi(a|s)=\mathrm{Pr}(a_t=a|s_t=s)$
$$
\begin{align}
v_{\pi}(s) &= \mathbb{E}[g_t(\tau))|s_t=s, \pi] \\
q_{\pi}(s,a) &= \mathbb{E}[g_t(\tau)|s_t=s, a_t = a, \pi]
\end{align}
$$  

Therefore the relationship between $v_{\pi}(s)$ and $q_{\pi}(s,a)$ can be presented as:  
$$
\begin{align}
v_{\pi}(s) &= \mathbb{E}[g_t(\tau)|s_t=s, \pi] \\ &= \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \mathbb{E}[g_t(\tau)|s_t=s, a_t = a, \pi] 
\\ &= \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \cdot q_{\pi}(s,a)
\end{align}
$$
"""


part1_q3 = r"""
**Your answer:**

<font color='darkgreen'>
1. First experiment: Result analysis
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_p</span> shows the policy-gradient as the negative average loss across trajectories.  
      Both $\hat\grad\mathcal{L}_{\text{PG}}(\vec{\theta})$ `vpg` and $\hat\grad\mathcal{L}_{\mathcal{H}(\pi)}(\vec{\theta})$ `epg` start from a high loss and gradually improve their policy (shown as loss values steadily climbing towards $0$).  
      Gradient derivations which subtract the baseline: $\hat\grad\mathcal{L}_{\text{BPG}}(\vec{\theta})$ `bpg` and $\hat\grad\mathcal{L}_{\text{CPG}}(\vec{\theta})$ `cpg` exhibit minimal fluctuations around $0$.  
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_e</span> shows the entropy loss as the negative entropy. The values rise as the networks learns with subsequent iterations. High entropy values imply that the action probability distribution resembles a uniform distribution, whereas a reduction in entropy marks the network's converges on the effective policy; the network is more confident of the proper actions.  
    The subtraction of baseline $b$ in $\hat\grad\mathcal{L}_{\text{CPG}}(\vec{\theta})$ `cpg` speeds up the rate of convergence (in comparison to $\hat\grad\mathcal{L}_{\mathcal{H}(\pi)}(\vec{\theta})$ `epg`) thus making the training process more efficient.
    * The graph <span style="font-family:Courier; font-size:1.2em;">baseline</span> illustrates that the baseline $b$ (the mean of batch q-values) increases with every batch. 
    $$\hat{q}_{t} = \sum_{t'\geq t} \gamma^{t'-t}r_{t'+1}.$$
    It is crucial for maintaining low variance that the magnitude of the baseline $b$ accounts for the increasing rewards.
    * The graph <span style="font-family:Courier; font-size:1.2em;">mean\_reward</span> shows an increase in mean reward, indicating that all evaluated losses were effective (to various degrees) for learning to solve the task. The introduction of entropy loss had little impact on the outcome of training, possibly due to the small action space. This conclusion stems from the similarity in mean reward between `epg` and `vpg` (no baseline, with and without entropy loss) and between `cpg` and `bpg` (baseline, with and without entropy loss). Also note that employing a baseline had a significant impact on the efficacy of the training process - manifesting in better mean rewards for `bpg` and `cpg`. 


2. Comparison between Regular Policy-Gradient `cpg` and Advantage Actor-Critic `AAC`
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_p</span> shows that `AAC` obtains a lower trajectory loss compared to `cpg` .  
        This is attributed to `AAC` being a more expressive model that better captures a reliable approximation of state values.
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_e</span> illustrates the entropy loss of `AAC` and `cpg` .  
    The two methods perform similarly - up to a scale factor determined by the ratio of entropy loss multipliers $~{}^{\mathbb{\beta}_{\text{CPG}}~~}{\mskip -5mu/\mskip -3mu}_{~~\mathbb{\beta}_{\text{AAC}}}$.
    * The graph <span style="font-family:Courier; font-size:1.2em;">mean\_reward</span> shows similar performance for `AAC` and `cpg` .  
    This can be attributed to the ability of the simple `cpg` (Policy-gradient with baseline) to reliably comprehend the task:  
    The task is simple enough as to not mandate the more expressive `AAC` approach.

"""
