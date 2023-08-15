# **Session 10: Revision**

## **Theory**
### **Question 1**
Explain why MDP planning is also called "temporally extended predictions into future, ahead of time" and what is "prediction" here?
### **Answer**

- MDP planning is also called “temporally extended predictions into future, ahead of time” because it involves looking ahead and estimating the expected outcomes of <mark>different courses of actions</mark> **(= extended)** <mark>over multiple time steps</mark> **(= temporally)**.
- The **prediction** here refers to the **expected value** or **utility of each state-action pair**, which are $V^\pi(s)$ and $Q^\pi(s,a)$: the mean discounted rewards in the future (after many courses of actions and state transitions) under policy $\pi$.
- For example, an agent playing chess can use MDP planning : The agent can predict the future states of the board and the rewards (win, lose or draw) based on its current state, its possible actions and the opponent’s responses. The agent can also <mark>**plan ahead for several moves in advance**</mark>, which is **temporally extended**.

### **Question 2**
Explain why MDP planning is also called "mental simulations"? What is "mental model" and what simulation, calculation is it used for? 
### **Answer**
- MDP planning is the "world model" including transition and reward function. Planner/solve use this model to obtain the next states & rewards and compute values (e.g with Bellman optimality equation). Every calculations can be performed simulatively without any real interactions with the environment $\rightarrow$ mental/model simulation.

### **Question 3**
Given a dataset with 2 trajectories (rollouts) as follows:
$$
𝝳_1 = S_3, A_1, S_1, A_2, S_2, A_1, 0.1, S_3, A_2, S_g, 1
$$

$$
𝝳_2 = S_1, A_2, S_1, -0.1, A_1, S_3, A_1, S_2, 0.2, A_2, S_3, A_2, S_g, 1
$$

- Calculate all return $G$, estimated state value $V$, action value $Q$ for states and actions if there is no discount ($\gamma = 1$)
- Calculate the **estimated** transition probabilities $P(s',s,a)$ and **average** reward $R(s,a)$ for states and values

### **Answer**
- $G^1(S_1) = 1.1$, $G^2(S_1) = 1.1$, $G^3(S_1)=1.2$
- $\hat{V}(S_1) = \frac{G^1 + G^2 + G^3}{3} \approx 1.13$
- There are two actions $\rightarrow$ $\hat{Q}(S_1, A_1) \approx 1.2$ and $\hat{Q}(S_1, A_2) \approx 1.1$
- Estimated transition probabilities $P(s',s,a)$ and average reward $R(s,a)$. Example for $S_1$ and $A_2$: 
    - $R(S_1, A_2) = \frac{0-0.1}{2} = -0.05$
    - $P(S_1, S_1, A_2) = P(S_2, S_1, A_2) = 0.5$ and the rest $P(S_3, S_1, A_2) = P(S_g, S_1, A_2) = 0$

## **Sumamry - MDP Planning (Perception - Action Loop)**
### **Planning is Sequential Decision Making**
- Planning is the core of intelligence: temporally extended predictions into future. Proper planning prevents poor performance. Fail to plan = plan to fail
- Motion-replanning in real-time with obstacles avoidance.
- Planning example of "Mental simulations" - "Thought experience" - "Planning ahead of time".
    -  Alpha zero
    -  Conversational AI

### **Markov Decision Process (MDP) Formulation**
- $M = (S, A, R, T, \gamma)$ with "world model" $(T,R)$ is used as "mental model" simulations: $s \xrightarrow{a} s' \sim r$
- Hypothesis: Reward is enough for intelligence
    - Objective: at each state $s$, find best action $a^*$ to maximize expected / average total / sum of future rewards
#### **Simplest formulation (Assumptions)**:
- One agent perceives its environment as $\bf o_t$ and take action $a_t$ at every time step $t$
- The agent can "fully observe" the environment "state": from observation $\bf o_t \rightarrow$ (internal) agent's state representation $\bf z_t =$ (external) environmental state $s_t$
-  Agent's $a_t$ causes $s_t$ to change in "Markovian" way $s_{t+1} \sim T(s_t, a_t)$ known to the agent.
-  The agent knows its task: at each and every situation, it gets a real number $r_{t+1} \sim R(s_t, a_t, s_{t+1})$ for reward or penalty/cost or whatever the task is done (either successfully or unsuccessfully) 

#### **Simplest formulation: $\text{MDP} = (S,A,T,R,\gamma)$**:
- A Markov Decision Process is 5-tuple $M = (S,A,T,R, \gamma)$ wit discount factor $0 \leq \gamma \leq 1$
- A planning algorithm $P$ solves the given MDP: $P(M) =$ a good plan which specifies a sequence of actions to efficiently achieve a goal

#### $(S,A,T,R,\gamma) \to  \text{policy } \pi(s) \text{ and value } q^\pi(s,a)$ 
- $P(M)$ = plan = a policy $\pi(s)$ having value $q^\pi(s,a)$ with $a \leftarrow \pi(s)$
    - What to do at each and every situation $\forall s \in S$: $a_t \leftarrow \pi(s_t)$
    - How good each action is in the future (sum of reward sense): action value $q^\pi(s,a)$ 
    - There are many ways / plannings / policy $\pi$ $\rightarrow$ Find the optimal one $\pi^*$
    $$
    \pi^* = \arg\max_\pi q^\pi(s,a), ~ \forall s\in S
    $$

### Trajectory, Episode, Return, Value, Optimal Policy
- A **trajectory** when executing a policy $\pi$ is the sequence of $s_t,a_t,s_{t+1}$ for $t=0$ to termination $T$. 
- The return of an eposide when executing a policy $\pi$ from state $s$ is the sum of discounted reward on the trajectory: 

$$
G^\pi (s) = \sum_{t=1}^T \gamma^{t-1} r_t \space \space | \space \space s_0 = s
$$

- The value of action $a$ at state $s$ is the average return of $K$ episodes when repeatedly take $a$ from $s$ & transit to $s' \sim T(s,a)$ then follow $\pi$ until termination:

$$
q^\pi(s,a) = \frac1{K} \sum_{k=1}^K \Big(r_1 + \gamma G^\pi_k(s') \Big)
$$

- An optimal policy $\pi^*$ has maximal action values at all states: 
$$
\pi^* = \arg\max_\pi q^\pi(s,a), ~ \forall s\in S
$$

### **Solving MDPs: Bellman Optimality Equation & Q-Value-Iteration**

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a)\max_{a'} Q^*(s',a')
$$

#### **Discrete finite MDP solver / planner**
- Dynamic Programming turns Bellman optimality equation into an iterative update rule:

$$\boxed{\begin{align}
&\text{Inputs: } S^+, A, \gamma \;\&\; R(s,a), P(s,a,s') \text{ arrays }\to \text{ Outputs: } \hat{Q}^*(s,a) \text{ and } \hat{\pi}^*(s)\\
\\
&\text{1. Initialize: }  Q_{0}(s,a) = 0, ~ \forall s,a\\
&\text{2. Iterate until termination: }\\
&\quad \forall s\neq\text{terminal states}, \forall a: ~ Q_{k+1}(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a)\max_{a'}(Q_k(s',a')).
\end{align}}$$