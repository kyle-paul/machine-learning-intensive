# Session 11: Revision
## **Theory**
### **Question 1**
Reinforcement learning is learn when we do not know $P(s'|s,a)$ and reward $R(r|s,a,s')$ functions by learning from the interaction data. In Q-learning, agent learn what and predict what?
### **Answer**

- Agent learn to predict optimal action value $Q^\pi(s,a)$ for all state-action pairs, which is the mean of all rewards if from state $s$, executing action $a$ under a policy $\pi$ until terminating. 
- In discrete MDP, the model of action value function $Q^\pi(s,a)$ is a table ($Q$ table) with each value in the table is a parameter need optimizing. Therefor, this table is also called "tabular Q-learning"
- We have the formula to update $Q(s,a)$ to reach the optimal $Q^\pi(s,a)$
$$
Q(s,a) \leftarrow Q(s,a) + \eta(r + \gamma \max_{a'} Q(s',a') - Q(s,a))
$$
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/1xg2ipllue5lQ86K8tQSjuN2TcAMFs2vEsH8eO1cjxG4FrwVGSwsUU3RFeHoam7i.png?d=desktop-thumbnail></center>

### **Question 2**
Explan what is behaviour policy $\pi_b$ & target/learning/optimizing plicy $\pi_o$ in Q-learning and the formula to compute $a = \pi(s)$.

### **Answer**
- Behaviour policy $\pi_b$ is policy for all actions interacting with the environment. Looking at the image above, at the first line of algorithm, $\pi_b$ can be $\epsilon -greedy$, or softmax, or even random.
- Target/Learning/optimizing policy $\pi_o$ is the policy for all actions to update Q-value (learning), which is line 3 $a = \arg \max Q(s,a)$ (greedy policy)

## **Summary**