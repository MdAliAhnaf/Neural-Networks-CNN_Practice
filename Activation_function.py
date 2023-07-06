#!/usr/bin/env python
# coding: utf-8

# # Activation Function

# Importing libraries in the environment for ipynb file execution

# In[17]:


import matplotlib.pyplot as plt
import numpy as np


# Setting the range for plotting graph for activation functions where step size is 1 and value is ranged between -10 to +10, and doing labelling onwards

# In[30]:


x = np.arange(-10, 11, 1)
print(x)
def plot_graph(y,ylabel):
    plt.figure()
    plt.plot(x,y, 'o--')
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel(ylabel)
    plt.show()


# Plotting Step Function's graph,
# step function is a mathematical function that takes on only two possible values, typically 0 or 1. It has a value of 0 for all negative inputs and a value of 1 for all positive inputs.

# In[31]:


y = list(map(lambda n: 1 if n>0.5 else 0, x))
plot_graph(y,"STEP(X)")


# Plotting sigmoid or logistic activation function's graph, it maps any input value to a value between 0 and 1, with an S-shaped curve where x is the input value and y is the output. 

# In[32]:


y = 1 / (1 + np.exp(-x))
plot_graph(y, "Sigmoid(X)")


# Plotting hyperbolic tangent(TanH) function's graph, it is a rescaled version of the sigmoid function, with outputs ranging between -1 and 1. With an S-shaped curve where x is the input value and y is the output.

# In[33]:


y = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
plot_graph(y, "TanH(X)")


# Plotting ReLU (Rectified Linear Unit) activation function's graph, It is a piecewise linear function that outputs the input directly if it is positive, and outputs 0 for any negative input. By Being linear for all positive inputs and non-linear for all negative inputs where x is the input value and y is the output.

# In[34]:


y = list(map(lambda a: a if a>=0 else 0, x))
plot_graph(y,"ReLU(X)")


# Plotting ELU (Exponential Linear Unit) activation function's graph, where x is the input value, alpha is a hyperparameter, and y is the output. It is a variation of the ReLU function and has been proposed as an alternative to it. The function outputs the input directly if it is positive, and has a smooth exponential transition for negative input values, which helps to prevent the "dying ReLU" problem, where neurons can become stuck in a negative output state.

# In[35]:


def elu(x, alpha=1.0):
   #return np.where(x >= 0, x, alpha*(np.exp(x)-1))
    return list(map(lambda a: a if a >= 0 else alpha * (np.exp(a) - 1), x))
y = elu(x)
plot_graph(y, "ELU(X)")


# Plotting SELU (Scaled Exponential Linear Unit) activation function's graph, where x is the input value, alpha and scale are hyperparameters, and y is the output. On the other hand of the elu function variation, the SELU function is a self-normalizing activation function that ensures that the output of each layer has zero mean and unit variance by using the two hyperparameters. This helps to reduce the vanishing and exploding gradient problems that can occur in deep neural networks.

# In[36]:


def selu(x, scale=1.0507, alpha=1.6733):
    #return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    return list(map(lambda a: scale * (a if a >= 0 else alpha * (np.exp(a) - 1)), x))
y = selu(x)
plot_graph(y, "SELU(X)")

