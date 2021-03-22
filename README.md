# Robust MuZero

A robust variant of MuZero

Authors: Masahiro Hayashi, Bakr Ouairem

Supervisors: [Edouard Leurent](http://edouardleurent.com/), [Odalric-Ambrym Maillard](http://odalricambrymmaillard.neowordpress.fr/)

# Overview

In order to propose a robust variant of MuZero, we first have to make the state transitions stochastic; to do so, we incorporate the work from PlaNet; namely, we integrate the SSM and RSSM model into the dynamics function of MuZero. We also modify MCTS so that the agent plans conservatively and robustly. So far, we haven't built a model that converges to any meaningful result.




