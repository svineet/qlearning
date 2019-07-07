Q Learning
==========

We estimate how good a state is by it's Q value.  
Then we select the states with high Q value.

Q values is learnt

Q maps (state, action) to a value

Q table is (bin, action) to value mapper  
that is ([val1, val2], action) to quality mapper

Doing that action in this state leads to good results
Is that the meaning of a q value is.

Instead of using a table, let's use a neural network

That network estimates Q given state and action
