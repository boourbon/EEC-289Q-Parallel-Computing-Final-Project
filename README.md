# EEC289Q_ParallelComputingFinalProject
In this project, we are going to apply a Reinforcement Learning DP algorithm on GPU. Dynamic Programming problems usually cannot be solved quickly using expensive software such as Matlab, and are also difficult to map to multi-core CPU implementations. Thus, we started to think about whether this kind of problem can be optimized for operational efficiency on the GPU, if we let enough threads to compute the subproblems of DP in parallel.
The specific dynamic programming problem we are trying to solve on the GPU this time is Jack's Car Rental problem on 'Reinforcement Learning: An Introduction' by Richard S.Sutton and Andrew G.Barto.

