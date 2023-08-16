# **Preclass Session 10: MDP Planning**

## **Markov Decision Process (MDP)**
### **Formulating MDP**
Let's see AI in application (a specific task): playing chess, self-driving car, household robots, chatbot. How can we optimize each **action/decision** of AI?
We can formulate the problem by sequential decisions/actions between AI **agent** and **environment** (e.g. chess, street, or external beings).

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/uOK1EMImvkvyPAZdcx6Q4OIwCWh1srWTSgTGGTqQOby5InJlPG7wzP8krGAxtg0b.png?d=desktop-thumbnail></center> <br>

- At time $t$, agent observes environment with the help of sensors (e.g. cameras) to gain ***observation*** $\bf o_t$ (e.g. an image of surrounding) and then extract feature to represent (percept) as ***state*** $s_t$. 
- Then, agent uses policy function $\pi(s)$ (input is state $s$) to compute the output action $a_t = \pi(s_t)$ which is the best action for the current state (this is calculated based on a particular performance measure).  
- When agent perform action $a_t$, the environment is transformed into state $s_{t+1}$. Then agent continues to observe $\bf o_{t+1}$ of state $s_{t+1}$.
- While this process continues, the agent can access the reward, or loss it receive when computes and performs each decision/action $a_t$ in response to each state $s_t$. Hence, this process is called **sequential decision making**. (or also called **experience snippet**)
$$
s_t \xrightarrow{a_t} s_{t+1} \approx r_{t+1} \space \text{or} \space s \xrightarrow{a} s' \approx r
$$

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/OJ55XCVR30tC8zPwX22mPjOMVl2Umk6ByW35DIuQXD08Bvm3h1Q1cjoakp4AHy4r.gif?d=desktop-thumbnail></center>

The MDP model will **simplify** the problem by
- Supposing that all history interactions (historry, context, memory, state as in RNN) can be represented as state $s_t$ without any loss of information. For example, when playing chess, we only consider the current state to make a decision, dismissing previous plays. This means we only consider **discrete state & action spaces**, which means $s \in S = \{S_1,...,S_N\}$ and $a \in A = \{A_1, ..., A_M\}$
The MDP will define another array as follows:
- **State transition function**: $s_{t+1} \approx P(s',s, a)$ with $s = s_t, a = a_t$. So, $P = [p_{ijk}$ is a 3-dimensional array $i = 1...N, j = 1...K, k = 1...N$ with values forming the probability vector $0 \leq p_{ijk} \leq 1$ and $\sum_i p_{ijk} = 1$
- **Rewward function**: $R(s,a,s')$ receives input $s_t, a_t, s_{t+1}$ and return a real number $r_{t+1} \approx R(s_t, a_t, s_{t+1})$

To emphasize the reward: we stipulate that the less time the agent get the reward, the more valued the reward will be. We define **Discount Factor** $0 \leq \gamma \leq 1$ to multiply with reward $r$ after each time-step. For example: wining a chess play will receive a reward $R = 1$ , but after 100 plays, the discounted reward $r_{100} = \gamma^{100}R$, which is significantly smaller than $r_{15} = \gamma^{15}R$ (winning after 15 steps).

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/bzMIM7WbYTFfE14ojZbN3L5JwlglDtuKqjyVZWGWEvRN1k3iuXCxu2muaUjyo5N6.png?d=desktop-thumbnail></center><br>


**Optimality**: what is the definition of an optimal decison/action: $a^* = \pi^*(s)$ at each state $s$. Note: because $a_t$ is computed from the policy function $\pi(s)$ cho each state $s_t$ as above $a_t = \pi(s_t)$.

- When agent interacts with the environment, we can terminate if the 
    1. Achieve the goal state $s_{t+1} \in S_{goal}$ 
    2. "Die" $s_{t+1} \in S_{terminal}$
    3. Time out $t = T_{max}$
- The period from which agent strrts and terminates is called a **episode** $i$ with the **trajectory** $\epsilon_i = (s_t, a_t, s_{t+1}, r_{t+1})_{t=0}^T$

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/RcFTyKrxDYrrlkTBlAlVwZkutmFlHs4m7ZpPcsn81Lsy9iun0bi2OXrkxpL7PNIB.png?d=desktop-thumbnail></center><br>


- The sum of discounted reward that agent collects along the trajectory $\epsilon_i$ of $i-th$ episode when beginning from 1 particular state $s_0 = S_k$ is called return $G_i(S_k) = \sum_{t=1}^T \gamma^t r_t$. Note: reward $r_t$ is often sparse (e.g. playing chess, only get $r_t \in (0, \pm 1)$ at the end of the game, the rest of game is absent, which means $r_t = 0$ most of the time in MDP).
- We want to maximize $G_i(s)$ for all episodes starting fom $s$, so we use mean value of $G_i(s)$ in $N_c$ episodes as performance metric, called state value function $V(s) = \frac{1}{N_c} \sum_{i=1}^{N_c} G_i(s)$
- If throughout the episode and trajectory, agent only follows 1 policy $\pi$, we will add to the formula $V^{\pi}$ and return $G_i^\pi(s)$ is under policy $\pi$
- To conveniently compute optimal policy, we define an **action value function** $Q^\pi(s, a) = R(s,a) + \gamma V^\pi (s')$, which is the mean return when starting from $s$, performing action $a$, and then following policy $\pi$ (with $R(s,a)$ is the mean reward at state $s$ when performing action $a$).
- Optimal action $a^*$ at state $s$ is $a^* = \pi^*(s) = argmax_a Q^*(s,a)$. And optimal policy is a policy which produces optimal actions at every state $\pi^* = argmax_\pi Q^\pi(s,a)$ with all $s \in S$

MDP planners/solvers can compute $\pi^*$ and $Q^*$ for all states and actions given transition function $P(s',s,a)$ and reward $R(s,a,s')$, called planning. If these two functions are not given beforehand, we cal unknown MDPs, agent need to learn from data when interactions, callled **Reinforcement Learning**.

### **Solving MDP: Planning methods**
- The algorithms for planning (MDP planning) are solvers for MDP model when given transition $P(s',s,a)$ and reward $R(s,a,s')$ functions. Bellman scientist proved the formula: **Bellman optimality equation** describing the reationship of 2 optimal values of 2 adjacent states.

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \space max_{a'} Q^*(s', a')
$$

- This equation can be be combined with **Dynamic Programmin**g with **Q-iteration**:
    <div style="border: 1px solid black;     padding: 10px;">
    
    Inputs: $S^+, A, \gamma, R(s,a), P(s, a, s')$
    
    Outputs: $\hat{Q}^*$ and $\hat{\pi}^*$ for all $s,a$
    
    1. Initialize: $Q_0(s,a) = 0$ with aol $s,a$
    2. Iterate until termination.
        With all $s \neq$ terminal states, with all $a$:
        $Q_{k+1}(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \space max_{a'} (Q_k(s', a'))$ 
    
</div>

- We can prove that this "dynamic-programming" loop will converge toward real value of optima value $Q^*(s,a)$ and optimal policy $\pi^*(s)$ for all pairs of $s,a$. (Planning for smalll MDPS will be covered in session 10)


### **Learning to control MDP: Reinforment Learning methods**
Given unknown MDPs devoid of components such as transition $P(s',s,a)$ or/and reward $R(s,a,s')$, we have to use machine learning methods (regression, classification, clustering,...) to approximately predict the optimal value $\hat{Q}^*(s,a)$ and optimal policy $\hat{\pi}^*(s)$ from snippets of data interactions.

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/ZlI9b74dH5bxu1Z7BXPhjuJze82NZcN5J478yXehE6Ty7pP0Tu0Qwcnu20XjRp7V.png?d=desktop-thumbnail></center><br>

#### **Small excercise**
- Given a basic operation model of a self-driving car composed of 3 states (Cool, Warm, Overheated) and two actions (Fast, Slow) as the image below:
    -  Let's tabulate the transition probabilites $P(s', s, a)$ and rewards $R(s,a,s')$
    -  Code to compute the optimal action value table $Q^*(s,a)$ to determine optimal decision (fast or slow) at each state.

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/UzN9udBBtPHMdqXPxXTXDt1gMKccM8X1RvMcfjLrHRrg0YkUmCgr243mnljz66vq.jpeg?d=desktop-thumbnail></center>