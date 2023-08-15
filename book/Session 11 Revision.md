## **Theory**
### **Question 1**
Reinforcement learning is learn when we do not know $P(s'|s,a)$ and reward $R(r|s,a,s')$ functions by learning from the interaction data. In Q-learning, agent learn what and predict what?
### **Answer**

- Agent learn to predict optimal action value $Q^\pi(s,a)$ for all state-action pairs, which is the mean of all rewards if from state $s$, executing action $a$ under a policy $\pi$ until terminating. 
- In discrete MDP, the model of action value function $Q^\pi(s,a)$ is a table ($Q$ table) with each value in the table is a parameter need optimizing.
- We have the formula to update $Q(s,a)$ to reach the optimal $Q^\pi(s,a)$
$$
Q(s,a) \leftarrow Q(s,a) + \eta(r + \gamma \max_{a'} Q(s',a') - Q(s,a))
$$

### **Question 2**
Behaviour policy $\pi$

### **Answer**

