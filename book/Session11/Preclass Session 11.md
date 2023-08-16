# **Preclass Session 11: Reinforcement Learning**

## **Definition**
Reinforcement learning (RL) is an area of machine learning that focuses on how you, or how some thing, might act in an environment in order to maximize some given reward. Reinforcement learning algorithms study the behavior of subjects in such environments and learn to optimize that behavior.

# **Part 1: Introduction to reinforcement learning**
## **Section 1: Markov Decision Processes (MDPs)**
### **Introduction to MDPs**
- Markov decision processes give us a way to formalize **sequential** decision making. This formalization is the basis for structuring problems that are solved with reinforcement learning.
- In an MDP, we have a decision maker, called an **agent**, that interacts with the environment it's placed in. These interactions occur sequentially over time. At each time step, the agent will get some representation of the environment's **state**. Given this representation, the agent selects an **action** to take. The environment is then transitioned into a new state, and the agent is given a **reward** as a consequence of the previous action.
- Components of an MDP: Agent, Environment, State, Action, Reward
- This process of selecting an action from a given state, transitioning to a new state, and receiving a reward happens sequentially over and over again, which creates something called a **trajectory** that shows the sequence of states, actions, and rewards.
- Throughout this process, it is the agent's goal to maximize the total amount of rewards that it receives from taking actions in given states. This means that the agent wants to maximize not just the immediate reward, but the **cumulative rewards** it receives over time.

### **MDP Notation**
- In an MDP, we have a set of states $\bf S$, a set of actions $\bf A$, and a set of rewards $\bf R$ . We'll assume that each of these sets has a finite number of elements.
- At each time step $t = 0,1,2,..$ the the agent receives some representation of the environment's state $S_t \in \bf S$. Based on this state, the agent selects an action $A_t \in \bf A$. This gives us the state-action pair $(S_t, A_t)$
- Time is then incremented to the next time step $t + 1$, and the environment is transitioned to a new state $S_{t+1} \in \bf S$. At this time, the agent receives a numerical reward $R_{t+1} \in \bf R$ for the action $A_t$ taken from state $S_t$. Let;s think this process as an arbitrary function $f$
$$
f(S_t, A_t) = R_{t+1}
$$
- The trajectory representing the sequential process of selecting an action from a state, transitioning to a new state, and receiving a reward can be represented as
$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3
$$

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/bzMIM7WbYTFfE14ojZbN3L5JwlglDtuKqjyVZWGWEvRN1k3iuXCxu2muaUjyo5N6.png?d=desktop-thumbnail></center><br>

- Let's break down this diagram into steps:
    1. At time $t$, the environment is in state $S_t$
    2. The agent observes the current state and selects action $A_t$
    3. The environment transitions into state $S_{t+1}$ and grants the agent reward $R_{t+1}$
    4. This process then starts over for the next time step $t + 1$
    - Note: $t + 1$ is no longer in the future, but is now the present. When we cross the dotted line on the bottom left, the diagram shows $t+1$ transforming into the current time step $t$ so that $S_{t+1}$ and $R_{t+1}$ are now $S_t$ and $R_t$ 

### **Transition probabilities**
- Since the sets $\bf S$ and $\bf R$ are finite, the random variables $R_t$ and $S_t$ have well defined probability distributions. In other words, all the possible values that can be assigned to $R_t$ and $S_t$ have some associated probability. These distributions depend on the preceding state and action that occurred in the previous time step $t-1$.
- For example, suppose $s' \in \bf S$ and $r \in \bf R$. Then there is some probability that $S_t = s'$ and $R_t = r$. This probability is determined by the particular values of the preceding state $s \in \bf S$ and action $a \in \bf A(s)$. Note that $\bf A(s)$ is the set of actions that can be taken from the state $s$.
- For all $s' \in {\bf S}, s\ in {\bf S}, r \in {\bf R}, a \in {\bf A(s)}$, we define the probability of transition to state $s'$ with reward $r$ from taking action $a$ in state $s$ as:
$$
p(s',r | s,a) = Pr\{S_t = s', R_t = r | S_{t-1} = s, A-{t-1} = a\}
$$

### **Expected return**
- We need a way to aggregate and formalize the cumulative rewards. For this, we introduce the concept of the expected return of the rewards at a given time step. For now, we can think of the return simply as the sum of future rewards. Mathematically, we define the return $\bf G$ at the time $t$:
$$
{\bf G_t} = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T
$$
- This concept of the expected return is super important because it's the agent's objective to maximize the expected return. The expected return is what's driving the agent to make the decisions it makes.

### **Episodic Vs. Continuing Tasks**
- In our definition of the expected return, we introduced $T$, the final time step. When the notion of having a final time step makes sense, the agent-environment interaction naturally breaks up into subsequences, called episodes. For example, think about playing a game of pong. Each new round of the game can be thought of as an episode, and the final time step of an episode occurs when a player scores a point.
- Each episode ends in a terminal state at time $T$, which is followed by resetting the environment to some standard starting state or to a random sample from a distribution of possible starting states. The next episode then begins independently from how the previous episode ended. Formally, tasks with episodes are called episodic tasks.
- There exists other types of tasks though where the agent-environment interactions don't break up naturally into episodes, but instead continue without limit. These types of tasks are called continuing tasks.
- Continuing tasks make our definition of the return at each time $t$ problematic because our final time step would be $\inf$. Because of this, we need to refine they way we're working with the return.

### **Discounted Return**
- Our revision of the way we think about return will make use of discounting. Rather than the agent's goal being to maximize the expected return of rewards, it will instead be to maximize the expected discounted return of rewards. Specifically, the agent will be choosing action $A_t$ at each time $t$ to maximize the expected discounted return.
- To define the discounted return, we first define the discount rate $\gamma$ to be a number between $0$ and $1$. The discount rate will be the rate for which we discount future rewards and will determine the present value of future rewards. With this, we define the discounted return as:
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} \\
= \sum_{k=0}^{\infty} \gamma^k R_{t + k + 1}
$$
- This definition of the discounted return makes it to where our agent will care more about the immediate reward over future rewards since future rewards will be more heavily discounted. So, while the agent does consider the rewards it expects to receive in the future, the more immediate rewards have more influence when it comes to the agent making a decision about taking a particular action.
- Now, check out this relationship below showing how returns at successive time steps are related to each other. We'll make use of this relationship later.
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} \\
= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4}) \\ 
= R_{t+1} + \gamma G_{t+1}
$$

- Also, check this out. Even though the return at time $t$ is a sum of an infinite number of terms, the return is actually finite as long as the reward is nonzero and constant, and $\gamma < 1$
- For example, if the reward at each time step is a constant $1$ and $\gamma < 1$, then the return is:
$$
G_t = \sum_{k=0}^{\infty} \gamma^k = \frac{1}{1 - \gamma}
$$

### **Policies and value functions**
- First, we'd probably like to know how likely it is for an agent to take any given action from any given state. In other words, what is the probability that an agent will select a specific action from a specific state? Secondly, in addition to understanding the probability of selecting an action, we'd probably also like to know how good a given action or a given state is for the agent. In terms of rewards, selecting one action over another in a given state may increase or decrease the agent's rewards, so knowing this in advance will probably help our agent out with deciding which actions to take in which states.
#### **Policy function**
- A policy is a function that maps a given state to probabilities of selecting each possible action from that state. We will use the symbol $\pi$ to denote a policy.
- When speaking about policies, formally we say that an agent “follows a policy.” For example, if an agent follows policy $\pi$ at time $t$, then $\pi(a|s) $is the probability that $A_t = a$ if $S_t = s$. This means that, at time $t$, **under policy $\pi$**, the probability of taking action $a$ in state $s$ is $\pi(a|s)$.
- For each state $s \in \bf S$, $\pi$ is a probability distribution over $a \in \bf A(s)$

#### **Value function**
- Value functions are functions of states, or of **state-action pairs**, that estimate how good it is for an agent to be in a given state, or how good it is for the agent to perform a given action in a given state.
- This notion of how good a state or state-action pair is is given in terms of expected return. Remember, the rewards an agent expects to receive are dependent on what actions the agent takes in given states. So, value functions are defined with respect to specific ways of acting. Since the way an agent acts is influenced by the policy it's following, then we can see that value functions are defined with respect to policies.
#### **State-Value Function**
- The state-value function for policy $\pi$, denoted as $v_{\pi}$, tells us how good any given state is for an agent following policy $\pi$. In other words, it gives us the value of a state under $\pi$. Formally, the value of state $s$ under policy $\pi$ is the expected return from starting from state $s$ at time $t$ and following policy $\pi$ thereafter. Mathematically we define $v_{\pi}(s)$ as:
$$
v_\pi(s) = E_\pi[G_t | S_t = s] \\
= E_\pi[\sum_{k=0}^\infty \gamma^k R_{t + k + 1} | S_t = s]
$$

#### **Action-Value Function**
- Similarly, the action-value function for policy $\pi$, denoted as $q_\pi$ tells us how good it is for the agent to take any given action from a given state while following policy $\pi$. In other words, it gives us the value of an action under $\pi$. 
- Formally, the value of action $a$ in state $s$ under policy $\pi$ is the expected return from starting from state $s$ at time $t$, taking action $a$ , and following policy $\pi$ thereafter. Mathematically, we define $q_\pi(s,a)$ as:

$$
q_\pi(s,a) = E_\pi [G_t | S_t = s, A_t = a] \\
= E_\pi[\sum_{k=0}^\infty \gamma^k R_{t + k + 1} | S_t = s, A_t = a]
$$

- Conventionally, the action-value function $q_\pi$ is referred to as the **Q-function**, and the output from the function for any given state-action pair is called a **Q-value**. The letter **“Q”** is used to represent the **quality** of taking a given action in a given state.

### **Learning optimal policies**
It is the goal of reinforcement learning algorithms to find a policy that will yield a lot of rewards for the agent if the agent indeed follows that policy. Specifically, reinforcement learning algorithms seek to find a policy that will yield more return to the agent than all other policies.
- In terms of return, a policy $\pi$ is considered to be better than or the same as policy $\pi'$ if the expected return of $\pi$ is greater than or equal to the expected return of $\pi'$ for all states.
$$
\pi > \pi' \leftrightarrow v_\pi(s) \geq v_\pi'(s) | s \in \bf S
$$
-  A policy that is better than or at least the same as all other policies is called the optimal policy.
#### **Optimal State-Value Function**
$$
\begin{equation*} v_{\ast }\left( s\right) =\max_{\pi }v_{\pi }\left( s\right) \end{equation*}
$$
- $v_*$ gives the largest expected return achievable by any policy $\pi$ for each state.
#### **Optimal Action-Value Function**
$$
\begin{equation*} q_{\ast }\left( s,a\right) =\max_{\pi }q_{\pi }\left( s,a\right) \end{equation*}
$$
-  $q_*$ gives the largest expected return achievable by any policy $\pi$ for each possible state-action pair.

### **Bellman Optimality Equation**
$$
\begin{eqnarray*} q_{\ast }\left( s,a\right) &=&E\left[ R_{t+1}+\gamma \max_{a^{\prime }}q_{\ast }\left( s^\prime,a^{\prime }\right)\right] \end{eqnarray*}
$$
- This is called the Bellman optimality equation. It states that, for any state-action pair $(s,a)$ at time $t$ , the expected return from starting in state $s$, selecting action $a$ and following the optimal policy thereafter (AKA the **Q-value** of this pair) is going to be the expected reward we get from taking action $a$ in state $s$, which is $R_{t+1}$, plus the maximum expected discounted return that can be achieved from any possible next state-action pair $(s',a')$
- Since the agent is following an optimal policy, the following state $s'$ will be the state from which the best possible next action $a'$ can be taken at time $t+1$
- We're going to see how we can use the Bellman equation to find $q_*$. Once we have $q_*$, we can determine the optimal policy because, with  $q_*$, for any state $s$, a reinforcement learning algorithm can find the action $a$ that maximizes $q_*(s,a)$



## **Section 2: Q-learning**
### **Introduction to Q-learning**
- Q-learning is the first technique we'll discuss that can solve for the optimal policy in an MDP.
- The objective of Q-learning is to find a policy that is optimal in the sense that the expected value of the total reward over all successive steps is the maximum achievable. So, in other words, the goal of Q-learning is to find the optimal policy by learning the optimal Q-values for each state-action pair.
### **Value iterations**
- The Q-learning algorithm iteratively updates the Q-values for each state-action pair using the Bellman equation until the Q-function converges to the optimal Q-function $q_*$. This approach is called value iteration. To see exactly how this happens, let's set up an example, appropriately called The Lizard Game.

![](https://hackmd.io/_uploads/rJ6UWj752.png)


| State | Reward |
| -------- | -------- | 
| One cricket | +1   |
| Empty |  - 1 
| Five crickets | +10 Game over |
| Bird|- 10 Game over|
-  The Q-values for each state-action pair will all be initialized to zero since the lizard knows nothing about the environment at the start. Throughout the game, though, the Q-values will be iteratively updated using value iteration.
-  As just mentioned, since the lizard knows nothing about the environment or the expected rewards for any state-action pair, all the Q-values in the table are first initialized to zero. Over time, though, as the lizard plays several episodes of the game, the Q-values produced for the state-action pairs that the lizard experiences will be used to update the Q-values stored in the Q-table.
- As the Q-table becomes updated, in later moves and later episodes, the lizard can look in the Q-table and base its next action on the highest Q-value for the current state. This will make more sense once we actually start playing the game and updating the table.

### **Storing Q-Values In A Q-Table**


| State | Left | Right | Up | Down |
| -------- | -------- | -------- | -------- | -------- |
| 1 Cricket     | 0    | 0  | 0 | 0|
| empty 1     | 0    | 0  | 0 | 0|
| empty 2     | 0    | 0  | 0 | 0|
| empty 3     | 0    | 0  | 0 | 0|
| Bird  | 0    | 0  | 0 | 0|
| empty 4      | 0    | 0  | 0 | 0|
| empty 5      | 0    | 0  | 0 | 0|
| empty 6      | 0    | 0  | 0 | 0|
| 5 crickets      | 0    | 0  | 0 | 0|

### **Exploration Vs. Exploitation**
- Exploration is the act of exploring the environment to find out information about it. Exploitation is the act of exploiting the information that is already known about the environment in order to maximize the return.
- The goal of an agent is to maximize the expected return, so you might think that we want our agent to use exploitation all the time and not worry about doing any exploration. This strategy, however, isn't quite right.
- Think of our game. If our lizard got to the single cricket before it got to the group of five crickets, then only making use of exploitation, going forward the lizard would just learn to exploit the information it knows about the location of the single cricket to get single incremental points infinitely. It would then also be losing single points infinitely just to back out of the tile before it can come back in to get the cricket again.
- If the lizard was able to explore the environment, however, it would have the opportunity to find the group of five crickets that would immediately win the game. If the lizard only explored the environment with no exploitation, however, then it would miss out on making use of known information that could help to maximize the return.

### **Implementing an epsilon greedy strategy**
- To get this balance between exploitation and exploration, we use what is called an epsilon greedy strategy. With this strategy, we define an exploration rate $\epsilon$ that we initially set to $1$. This exploration rate is the probability that our agent will explore the environment rather than exploit it. With $epsilon = 1$, it is $100%$ certain that the agent will start out by exploring the environment.
- As the agent learns more about the environment, at the start of each new episode, $\epsilon$ will decay by some rate that we set so that the likelihood of exploration becomes less and less probable as the agent learns more and more about the environment. The agent will become **“greedy”** in terms of exploiting the environment once it has had the opportunity to explore and learn more about it $\rightarrow$ **epsilon decay and greedy policy**
- To determine whether the agent will choose exploration or exploitation at each time step, we generate a random number between $0$ and $1$. If this number is greater than $\epsilon$, then the agent will choose its next action via exploitation, i.e. it will choose the action with the highest Q-value for its current state from the Q-table. Otherwise, its next action will be chosen via exploration, i.e. randomly choosing its action and exploring what happens in the environment. See thw pseudocode:
```python
if random_num > epsilon:
# choose action via exploitation
else:
# choose action via exploration
```
### **Updating The Q-Value**
#### **Calculating Loss**
- To update the Q-value for the action of moving right taken from the previous state, we use the Bellman equation that we highlighted previously:
$$
\begin{eqnarray*} q_{\ast }\left( s,a\right) &=&E\left[ R_{t+1}+\gamma \max_{a^{\prime }}q_{\ast }\left( s^\prime,a^{\prime }\right)\right] \end{eqnarray*}
$$
- We want to make the Q-value for the given state-action pair as close as we can to the right hand side of the Bellman equation so that the Q-value will eventually converge to the optimal Q-value $q_*$
- This will happen over time by **iteratively** comparing the loss between the Q-value and the optimal Q-value for the given state-action pair and then updating the Q-value over and over again each time we encounter this same state-action pair to **reduce the loss**.
$$
\begin{eqnarray*} q_{\ast }\left( s,a\right) - q(s,a)&=&loss \\E\left[ R_{t+1}+\gamma \max_{a^{\prime }}q_{\ast }\left( s^\prime,a^{\prime }\right)\right] - E\left[ \sum_{k=0}^{\infty }\gamma ^{k}R_{t+k+1}\right]&=&loss \end{eqnarray*}
$$

#### **The Learning Rate**
- The learning rate is a number between $0$ and $1$, which can be thought of as how quickly the agent abandons the previous Q-value in the Q-table for a given state-action pair for the new Q-value.
- We don't want to just overwrite the old Q-value, but rather, we use the learning rate as a tool to determine how much information we keep about the previously computed Q-value for the given state-action pair versus the new Q-value calculated for the same state-action pair at a later time step. We'll denote the learning rate with the symbol $\alpha$, and we'll arbitrarily set 
$\alpha=0.7$ for our lizard game example, which means we want to get only $70$% of new information.
- The higher the learning rate, the more quickly the agent will adopt the new Q-value. For example, if the learning rate is $1$, the estimate for the Q-value for a given state-action pair would be the straight up newly calculated Q-value and would not consider previous Q-values that had been calculated for the given state-action pair at previous time steps.

#### **Calculating The New Q-Value**
$$
\begin{equation*} q^{new}\left( s,a\right) =\left( 1-\alpha \right) ~\underset{\text{old value} }{\underbrace{q\left( s,a\right) }\rule[-0.05in]{0in}{0.2in} \rule[-0.05in]{0in}{0.2in}\rule[-0.1in]{0in}{0.3in}}+\alpha \overset{\text{ learned value}}{\overbrace{\left(
                                        R_{t+1}+\gamma \max_{a^{^{\prime }}}q\left( s^{\prime },a^{\prime }\right) \right) }} \end{equation*}
$$
- So, our new Q-value is equal to a weighted sum of our old value and the learned value. The old value in our case is $0$ since this is the first time the agent is experiencing this particular state-action pair, and we multiply this old value by $(1-\alpha)$
- Our learned value is the reward the agent receives from moving right from the starting state plus the discounted estimate of the optimal future Q-value for the next state-action pair $(s',a')$ at time $t+1$. This entire learned value is then multiplied by our learning rate.
- All of the math for this calculation of our concrete example state-action pair of moving right from the starting state is shown below. Suppose the discount rate $\gamma=0.99$. We have:

$$
\begin{eqnarray*} q^{new}\left( s,a\right) &=&\left( 1-\alpha \right) ~\underset{\text{old value}}{\underbrace{q\left( s,a\right) }\rule[-0.05in]{0in}{0.2in} \rule[-0.05in]{0in}{0.2in}\rule[-0.1in]{0in}{0.3in}}+\alpha \overset{\text{ new value}}{\overbrace{\left(
                                        R_{t+1}+\gamma \max_{a^{^{\prime }}}q\left( s^{\prime },a^{\prime }\right) \right) }} \\ &=&\left( 1-0.7\right) \left( 0\right) +0.7\left( -1+0.99\left( \max_{a^{^{\prime }}}q\left( s^{\prime },a^{\prime
                                        }\right) \right) \right) \end{eqnarray*}
$$

- Let's pause for a moment and focus on the term $\max_{a^{^{\prime }}}q\left( s^{\prime },a^{\prime }\right)$. Since all the Q-values are currently initialized to $0$in the Q-table, we have:

$$
\begin{eqnarray*} \max_{a^{^{\prime }}}q\left( s^{\prime },a^{\prime }\right) &=&\max \left( q\left( \text{empty6, left}\right),q\left( \text{empty6, right}\right),q\left( \text{empty6, up}\right),q\left( \text{empty6, down}\right) \right) \\
                                        &=&\max \left( 0\rule[-0.05in]{0in}{0.2in},0,0,0\right) \\ &=&0 \end{eqnarray*}
$$
- Now, we do the math:
$$
\begin{eqnarray*} q^{new}\left( s,a\right) &=&\left( 1-\alpha \right) ~\underset{\text{old value}}{\underbrace{q\left( s,a\right) }\rule[-0.05in]{0in}{0.2in} \rule[-0.05in]{0in}{0.2in}\rule[-0.1in]{0in}{0.3in}}+\alpha \overset{\text{ new value}}{\overbrace{\left(
                                        R_{t+1}+\gamma \max_{a^{^{\prime }}}q\left( s^{\prime },a^{\prime }\right) \right) }} \\ &=&\left( 1-0.7\right) \left( 0\right) +0.7\left( -1+0.99\left( \max_{a^{^{\prime }}}q\left( s^{\prime },a^{\prime
                                        }\right) \right) \right) \\ &=&\left( 1-0.7\right) \left( 0\right) +0.7\left( -1+0.99\left( 0\right) \right) \\ &=&0+0.7\left( -1\right) \\ &=&-0.7 \end{eqnarray*}
$$
- Alright, so now we'll take this new Q-value we just calculated and store it in our Q-table for this particular state-action pair.
- Once the Q-function converges to the optimal Q-function, we will have our optimal policy. From now then, the agent only need to **follow the optimal policy** $\pi$ which is the value $q_*(s,a)$ in the $Q$ table.
- 

## **Section 3: Code project - Implement Q-learning with pure Python to play a game**
[Read gym open AI](https://openai.com/research/openai-gym-beta)
[Jupyter notebook for this game](https://colab.research.google.com/drive/1U-AVKMqYghfIWrEpjAtrRJSCCZT4uX7y?usp=sharing)
- Environment set up and intro to OpenAI Gym
- Write Q-learning algorithm and train agent to play game
- Watch trained agent play game

# **Part 2: Deep reinforcement learning**
### **Section 1: Deep Q-networks (DQNs)**
- Introduction to DQNs
- Replay Memory Explained
Training a Deep Q-Network
- TTraining a DQN With Fixed Q-Targets
## **Section 2: Code project - Implement deep Q-network with PyTorch**
- Deep Q-Network Code Project Intro
- Build Deep Q-Network in Code
- DQN Image Processing And Env Management
- Deep Q-Network Training Code
- Solving Cart and Pole With a DQN