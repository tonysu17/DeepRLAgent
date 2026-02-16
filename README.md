# DeepRLAgent
High-fidelity market simulation. Deep RL agent. 

This project involves two components: (1) a sophisticated RL trading agent inspired by Deepstack / Alphago, and (2) the environment in which the agent operates in. I would like to additionally incorporate counterfactual regret minimisation (CFR) as part of the network, where the CFR serves as a means to remain unexploitabilitable with regards to adversary opponents, and to better guide search. Over iterations of self-play, the CFR will converge to a Nash equilibrium for a certain scenario, then the Nash equilibrium will guide and refine the search process. 

## RL Agent
The agent utilises an actor-critic architecture. The main innovation is to use the value network as an "intuition" estimator, which is directly inspired by the same mechanism in game playing algorithms, and is used to assess the quality of future states conditoned on the current state. 

## High-fidelity market environment 
