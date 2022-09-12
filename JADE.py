"""
Implementation of the adaptative DE algorithm proposed by Zhang, J. et al, 
JADE: Adaptive Differential Evolution with Optional External Archive.

Author: Antoine Belley
Date: August 2022
"""

import numpy as np
from numpy.random import default_rng

rng = default_rng()


class JADE:

  
  def __init__(self,  dim, bounds=None):
    self.u_F = 0.6
    self.u_CR = 0.5
    self.dim = dim
    self.bounds = bounds

  
  def initialize_population(self, NP):  
    population = np.random.rand(NP, self.dim)
    population = self.scale_bound(population, NP)
    return self.apply_bounds(population)

  def scale_bound(self, pop, NP):
    minimum = [bound[0] for bound in self.bounds]
    maximum = [bound[1] for bound in self.bounds]
    for i in range(NP):
      pop[i,:] = (1-pop[i,:])*minimum + maximum*pop[i,:]
    return pop
  

  def apply_bounds(self, pop):
    if self.bounds is None:
      return pop
    else: 
      minimum = [bound[0] for bound in self.bounds]
      maximum = [bound[1] for bound in self.bounds]
      return np.clip(pop, minimum, maximum)


  def mutation(self, population, pop_score, f, p):
    """Apply the mutation strategy DE/current-to-p-best used
    in the JADE algorithm"""

    NP = population.shape[0]
    #Find best parent
    p_best =[]
    for p_i in p:
        best_index = np.argsort(pop_score)[:max(2, int(round(p_i*len(population))))]
        p_best.append(rng.choice(best_index))
    p_best = np.array(p_best)
    #Pick to other random parents
    choices = np.indices((NP, NP))[1]
    mask = np.ones(choices.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    choices = choices[mask].reshape(NP, NP - 1)
    parents_index = np.array([rng.choice(row, 2, replace=False) for row in choices])
    par1 = population[parents_index[:,0]]
    par2 = population[parents_index[:,1]]
    mutated = population + f * (population[p_best] - population + par1 - par2)
    return self.apply_bounds(mutated)
    
      
  def crossover(self,  population, mutated, cr):
    """Binary crossover from previous gen to new one"""
    chosen = np.random.rand(*population.shape)
    j_rand = rng.integers(low=0, high=population.shape[1])
    chosen[j_rand::population.shape[1]] = 0
    return np.where(chosen <= cr, mutated, population)


  def run(self, function,  NP=10, max_evals = 1000, c=0.1):
    population = self.initialize_population(NP)
    score = np.array( [ function(  population[i] ) for i in range(NP) ] )
    p  = np.ones(NP)* max(.05, 3/NP)
    for G in range(max_evals):
      print(G)
      #Parameters for current generation
      cr = rng.normal(self.u_CR, 0.1,NP)
      f = np.random.rand(NP // 3) * 1.2
      f = np.concatenate((f, rng.normal(self.u_F, 0.1, NP - (NP// 3))))

      #Evolution steps
      mutated = self.mutation(population, score,f.reshape(len(f),1),  p)
      crossed = self.crossover(population, mutated, cr.reshape(len(cr),1))
      c_score = np.array( [ function(  crossed[i] ) for i in range(NP) ] )
      
      #Select the new values
      indexes = np.where(score > c_score)[0]
      population[indexes] = crossed[indexes]
      score[indexes] = c_score[indexes]

      #Adapt for next step
      if len(indexes) != 0:
        self.u_CR = (1 - c) * self.u_CR + c * np.mean(cr[indexes])
        self.u_F = (1 - c) * self.u_F + c * (np.sum(f[indexes]**2) / np.sum(f[indexes]))
    
    index = np.argmin(score)
    return population[index], score[index]