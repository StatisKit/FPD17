#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:34:40 2020

@author: durand
"""

# An Application to Mango Patchiness Analysis

import matplotlib
from matplotlib import pyplot

from statiskit import (linalg,
                       core)

from statiskit.data import core as data
data = data.load('FPD18')

# conversion to pandas.dataframe
data.write_csv("file.csv")

import numpy as np
t = np.loadtxt("file.csv", dtype=np.int64)

print("max sum: " + str(max(t.sum(1))))
print("min sum: " + str(min(t.sum(1))))
print("max: " + str(t.max()))

"""Multinomial splitting distributions of mixture components were of dimension 
$3$, where $N_0$ (resp. $N_1$ and $N_2$) denotes the number of vegetative 
(resp. reproductive and quiescent) patches observed within a tree.
Only the multinomial and the Dirichlet multinomial  were considered for the 
singular distribution.
Within each component the parametric form of the singuar distribution was 
selected using the BIC.
"""

singular_estimator = core.singular_selection('MN', 'DM')

""" Since there is at least one patch in a mango tree (i.e., the tree itself), 
shifted singular multinomial splitting distributions were considered with a 
shift equal to $1$ for binomial, negative binomial and Poisson sum distributions 
but without shift for geometric and logarithmic distributions.
Within each component the parametric form of the sum distribution was selected 
using the BIC.
"""

# As a reminder: univariate discrete distributions referred to in our article:
# Binomial, negative binomial, Poisson, logarithmic series, Beta binomial, 
# Beta negative binomial, Beta Poisson

# Available here: 
# 'PoissonDistribution', 'BinomialDistribution', 'LogarithmicDistribution', 
# 'GeometricDistribution', 'NegativeBinomialDistribution', 
# 'BetaBinomialDistribution', 'BetaNegativeBinomialDistribution',

# Now singular discrete multivariate:
# Article: Multinomial, Dirichlet multinomial, multivariate hypergeometric,
# generalized Dirichlet multinomial (asymmetric), logistic normal multinomial

# Available here:
# DirichletMultinomialSingularDistribution
# MultinomialSingularDistribution

# Can multivariate hypergeometric fit the data? Check admissible values for parameters and support.

# generalized Dirichlet multinomial: check marginals and constraints on
# parameters and data. See also:
# N. Bouguila, "Clustering of Count Data Using Generalized Dirichlet Multinomial Distributions,"
# in IEEE Transactions on Knowledge and Data Engineering, vol. 20, no. 4, pp. 462-474, April 2008, doi: 10.1109/TKDE.2007.190726.
sum_estimator = core.shifted_estimation(core.outcome_type.DISCRETE,
                                        shift = 1,
                                        estimator = core.selection(core.outcome_type.DISCRETE,
                                                                   estimators = [core.poisson_estimation('ml'),
                                                                                 core.binomial_estimation('ml'),
                                                                                 core.negative_binomial_estimation('ml')]))
# The role of core.geometric_estimation('ml'), core.logarithmic_estimation('ml')
# is not clear

sum_estimator = core.selection(core.outcome_type.DISCRETE,
                               estimators = [sum_estimator,
                                             core.geometric_estimation('ml'),
                                             core.logarithmic_estimation('ml')])

"""
With `singular_estimator` and `sum_estimator` we are now able to construct 
an estimator for a splitting distribution.
"""

estimator = core.splitting_estimation(sum = sum_estimator,
                                      singular = singular_estimator)
"""
The initilization of the EM alogrithm is made with a mixture of $27$ components 
with slight differences concerning multinomial singular distribution probabilities.
"""

import itertools
pi = [1.] * len(data.components)
components = []
# product set {1., 2., 3.}^3
# alpha is the parameter of a singular multinomial distribution and is normalized
# in the constructor
# The weights of the components are 1. (unnormalized) -> 1/27
# The components can be seen with "print(components)"
for alpha in itertools.product(*([(1.,2.,3.)] * 3)):
    components.append(core.SplittingDistribution(core.PoissonDistribution(1.),
                                                 core.MultinomialSingularDistribution(linalg.Vector(alpha))))
initializator = core.MixtureDistribution(*components,
                                         pi = linalg.Vector([1.] * len(components)))
mixest = [core.mixture_estimation(data, 'em',
                                  initializator = initializator,
                                  default_estimator = estimator)]

"""
We then estimate the mixture models with an initialization were components less observed *a posteriori* are successively removed.
"""

# Obviously here we remove elements from "components"
while len(components) > 2:
    # Count each restored state
    assignment = list(mixest[-1].estimated.assignment(data))
    components.pop(min(set(assignment), key=assignment.count))
    initializator = core.MixtureDistribution(*components,
                                             pi = linalg.Vector([1.] * len(components)))
    mixest.append(core.mixture_estimation(data, 'em',
                                           initializator = initializator,
                                           default_estimator = estimator, maxits=100))

"""
We also considered the case where there is only one component.
"""

mixest.append(estimator(data, False))
mixest = list(reversed(mixest))

# TODO: check what is actually mixed. Sum distributions, splitting, or both (with different weights then)?
# Or is it rather a mixture of splitting as it rather seems?

"""
To select the best number of components, we used the BIC.
"""

import math
Sm = [result.estimated.loglikelihood(data) for result in mixest]
Dm = [mixest[0].estimated.nb_parameters] + [result.estimated.nb_parameters for result in mixest[1:]]
BICm = [s - d * math.log(data.total) / 2 for s, d in zip(Sm, Dm)]
Cm = [1] + [result.estimated.nb_states for result in mixest[1:]]
# entropy
Um = [0] + [result.estimated.uncertainty(data) for result in mixest[1:]]
ICLm = [bic - u for (bic, u) in zip(BICm, Um)]
limit = 10
fig = pyplot.figure(figsize=(3,3))
axes = fig.add_subplot(111)
axes.plot(Cm[:limit], ICLm[:limit], color='k')
axes.set_xticks(range(1, limit + 1))
axes.set_xlabel('Number of mixture components')
axes.set_ylabel('ICL')
axes.grid(linestyle='--')
pyplot.tight_layout()
try:
    fig.savefig('selection.pgf')
except:
    pass
fig.savefig('selection.svg')

"""
The best number of components is $2$ with relatively similar representation
"""

mixest[1].estimated.pi

"""
In component $1$, the best observation distribution is a multinomial splitting shifted negative binomial.
"""

mixest[1].estimated.observations[0]

"""
This is also true in component $2$
"""

mixest[1].estimated.observations[1]

"""
To appreciate the fit of this model, on can look at the fit of the sum distributions for example.
"""

sum_data = core.from_list([sum([event.value for event in events]) for events in data.events])
dist = mixest[1].estimated
fig = pyplot.figure(figsize=(6,3))
sum_dist = core.MixtureDistribution(dist.observations[0].sum,
                                    dist.observations[1].sum,
                                    pi = dist.pi)
axes = sum_dist.pdf_plot(axes = fig.add_subplot(111),
                         color = 'k',
                         fmt = '-',
                         norm = sum_data.total,
                         qmin = sum_data.min.value,
                         qmax = sum_data.max.value,
                         states = [dict(fmt = ':'),
                                   dict(fmt = '--')])
axes = sum_data.pdf_plot(axes = axes, color='#e6e6e6',
                                      edgecolor='k')
axes.set_xlabel('n')
axes.set_ylabel('Frequency')
pyplot.tight_layout()
try:
    fig.savefig('sum.pgf')
except:
    pass
fig.savefig('sum.svg')
"""
Note that if we consider only $1$ component, the best model is a Dirichlet multinomial splitting geometric distribution.
"""

mixest[0].estimated

print("Mean for component 0 sum: " + str(dist.observations[0].sum.mean))
print("Mean for component 1 sum: " + str(dist.observations[1].sum.mean))

# numpy conversion of proportions
np_rho = np.array([dist.pi[i] for i in range(dist.pi.nb_rows)])

# Chi-square goodness of fit test

e = data.events

e_counts = {} # count occurrences of event
e_probs = {} # associated probabilities
e_counts_keys = []
for i in range(len(e)):
    k = repr(e[i])
    if k in e_counts_keys:
        e_counts[k] += 1
    else:
        e_counts_keys += [k]
        e_counts[k] = 1
        e_probs[k] = dist.probability(e[i])

# Obviously probabilities are wrong
# and so is the max log-likelihood, see below
print("Sum of probabilities: " + str(np.sum(list(e_probs.values()))))

def get_nb_parameters(nb):
    """
    get parameters of a negative binomial distribution    
    """
    s = str(nb.probability)
    i1 = s.find('NB')
    i2 = s.find(',', i1)
    r = float(s[i1+3:i2])
    i3 = s.find(',', i2+1)
    p = 1.-float(s[i2+1:i3]) # the parameterisation is p'=1-p in scipy
    i4 = s.find(')', i3+1)
    d = int(s[i3+1:i4])
    return (r, p, d)

def probability(y, pi, r, p, delta):
    """
    probability computation for a splitting multinomial - negative binomial 
    distribution 
    """
    from scipy.stats import multinomial, nbinom
    s = y.sum() - delta
    if s >=0:
        m = multinomial(y.sum(), pi) 
        b = nbinom(r, p, delta) 
        pr = m.pmf(y) * b.pmf(y.sum())
        pr = pr[0]
    else:
        pr = 0.0
    return(pr)

pi = np.zeros((len(np_rho), 3))

for i in range(pi.shape[0]):
    for j in range(pi.shape[1]):
        pi[i,j] = dist.observations[i].singular.pi[j]

r = np.zeros((len(np_rho), 1))
p = np.zeros((len(np_rho), 1))
d = np.zeros((len(np_rho), 1), dtype=np.int64)

for i in range(pi.shape[0]):
    r[i], p[i], d[i] = get_nb_parameters(dist.observations[i].sum)


# Marginal means in each component
dist.observations[0].sum.mean * pi[0,:]
dist.observations[1].sum.mean * pi[1,:]

def event_to_array(e):
    """
    Conversion from event to numpy.array
    """
    return (np.array(eval(repr(e))))

probs1 = {}

for k in e_probs.keys():
    probs1[eval(k)] = probability(np.array(eval(k)), pi[0,:], r[0], p[0], d[0])
    
np.array(list(probs1.values())).sum()

def mixture_probability(y, rho, pi, r, p, delta):
    """
    probability computation for a mixture of splitting multinomial 
    - negative binomial distributions
    """
    k = len(rho) # number of mixture components
    p = [probability(y, pi[i], r[i], p[i], delta[i]) for i in range(k)]
    p = np.array(p)
    w = np.array(rho)
    return (np.dot(p, w))

probs = {}

for k in e_probs.keys():
    probs[eval(k)] = mixture_probability(np.array(eval(k)), np_rho, pi, r, p, d)
    
cov_prob = np.array(list(probs.values())).sum()

full_mixture_probs = {}
for v in itertools.product(*([range(12)] * 3)):
    if np.sum(v) > 0:
        full_mixture_probs[v] = mixture_probability(np.array(v), np_rho, pi, r, p, d)

np.array(list(full_mixture_probs.values())).sum()    

# Complement to 1:
comp = 1 - np.array(list(full_mixture_probs.values())).sum()

# Find probable values not occurring in data
[(k,v) for (k,v) in full_mixture_probs.items() if v > 0.005 and k not in probs.keys()]

# Aggregate keys / cells
m_keys = list(full_mixture_probs.keys())
nb_m_keys = len(m_keys)
# Mark a key when used in aggregation
mark_m_keys = dict(zip(m_keys, [False]*nb_m_keys))
current_m_k = 0
e_counts_cp = dict([(eval(k), v) for k, v in e_counts.items()])
total_counts = np.array(list(e_counts.values())).sum()
e_counts_n = {}
e_probs_n = {}
for k in e_counts_cp.keys():
    if not(mark_m_keys[k]) and ((e_counts_cp[k] < 6) or \
        (total_counts * full_mixture_probs[k] < 5)):
        current_count = e_counts_cp[k]
        current_e_freq = total_counts * full_mixture_probs[k]
        current_key = [k]
        # enlarge cell
        assert(not(mark_m_keys[k]))
        mark_m_keys[k] = True
        while current_m_k < nb_m_keys and (current_count < 6 or \
            current_e_freq < 5):
            # append free cells in m_keys            
            mk = m_keys[current_m_k] # candidate
            if not(mark_m_keys[mk]):
                if mk in e_counts_cp:
                    current_count += e_counts_cp[mk]
                    current_e_freq += total_counts * full_mixture_probs[mk]
                assert(not(mark_m_keys[mk]))    
                mark_m_keys[mk] = True
                # Append new key
                current_key += [mk]
            current_m_k += 1            
        if current_m_k < nb_m_keys:
            e_counts_n[str(current_key)] = current_count
            e_probs_n[str(current_key)] = current_e_freq
        else:
            # unmark all event components
            for v in current_key:
                mark_m_keys[v] = False
                e_counts_n.pop(v, None)
                e_probs_n.pop(v, None)
    elif not(mark_m_keys[k]):
        # Mark key
        assert(not(mark_m_keys[k]))
        mark_m_keys[k] = True
        e_counts_n[str([k])] = e_counts_cp[k]
        e_probs_n[str([k])] = total_counts * full_mixture_probs[k]

# group remaining cells

current_key = []
current_count = 0
current_e_freq = 0.
for k in e_counts_cp.keys():
    if not(mark_m_keys[k]):
        current_key += [k]
        current_e_freq += total_counts * full_mixture_probs[k]
        current_count += e_counts_cp[k]
        mark_m_keys[k] = True
if str(current_key) != '[]':
    e_counts_n[str(current_key)] = current_count
    e_probs_n[str(current_key)] = current_e_freq

np.array(list(e_counts_n.values())).sum() # total counts

# Check key unicity and Compute associated probabilities
mark_m_keys_check = dict(zip(m_keys, [False]*nb_m_keys))
se = [] # events involved in keys in e_counts_n

for k in e_counts_n.keys():
    e_probs_n[k] = e_probs_n[k] / total_counts
    vals = eval(k)
    for v in vals:
        assert(not(mark_m_keys_check[v]))
        mark_m_keys_check[v] = True
        se += [v]

assert(abs(np.array(list(e_probs_n.values())).sum() - cov_prob) < 1e-10) # total expected counts
assert(len(set(e_counts_cp.keys()).difference(set(se)))==0)
assert(np.array(list(e_counts_n.values())).sum() == total_counts)

# Difference of event sets
assert(str(e_counts_n.keys()) == str(e_probs_n.keys()))
assert(len(set(se).difference(set(full_mixture_probs.keys()))) == 0)
set(full_mixture_probs.keys()).difference(set(se))
assert(sum(np.array(list(mark_m_keys.values())) == False) == \
       len(set(full_mixture_probs.keys()).difference(set(se)) ))


    
# Add remaining cells in full_mixture_probs to min prob cell
rem_fmp = [k for k in full_mixture_probs.keys() if not(mark_m_keys[k])]
rem_fmp_prob = np.array([full_mixture_probs[k] for k in rem_fmp]).sum()

m = 1.
for (k, v) in e_probs_n.items():
    if v < m:
        min_k = k
        m = v

min_k_n = str(eval(min_k) + rem_fmp)

v = e_counts_n.pop(min_k, None)
e_counts_n[min_k_n] = v

e_probs_n.pop(min_k, None)
e_probs_n[min_k_n] = m + rem_fmp_prob

# Add complement wrt full_mixture_probs to min prob cell
m = 1.
for (k, v) in e_probs_n.items():
    if v < m:
        min_k = k
        m = v

min_k_n = min_k[0:-1] + ", Rem]"

v = e_counts_n.pop(min_k, None)
e_counts_n[min_k_n] = v

e_probs_n.pop(min_k, None)
e_probs_n[min_k_n] = m + comp


from scipy.stats import chisquare

f_obs = [] # observed nb counts
f_exp = [] # expected nb counts
for k, v in e_counts_n.items():
    f_obs += [v]
    f_exp += [e_probs_n[k]]
    
f_obs = np.array(f_obs)    
f_exp = np.array(f_exp) 
total_counts = f_obs.sum()
f_exp = total_counts * f_exp

contribs = {}
i = 0
for k, v in e_counts_n.items():
    contribs[k] = ((f_obs[i] - f_exp[i])**2 / f_exp[i], \
                   np.sign((f_obs[i] - f_exp[i])), f_obs[i], f_exp[i])
    i += 1
    
ddof = dist.nb_parameters
chisquare(f_obs, f_exp=f_exp, ddof=ddof)

# p-value
chi = np.array([c[0] for c in contribs.values()]).sum()
from scipy.stats import chi2 
pv = 1-chi2.cdf(chi, len(contribs) - 1 - ddof)

# Check log-likelihood

dl = dist.loglikelihood(data)

ml = 0.

for (k,v) in e_counts_cp.items():
    ml += np.log(mixture_probability(np.array(k), np_rho, pi, r, p, d)) * v 
    
def my_icl(dist, data):
    """
    Alternative computation of ICL
    """
    np_rho = np.array([dist.pi[i] for i in range(dist.pi.nb_rows)])

    pi = np.zeros((len(np_rho), 3))

    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            pi[i,j] = dist.observations[i].singular.pi[j]
    
    r = np.zeros((len(np_rho), 1))
    p = np.zeros((len(np_rho), 1))
    d = np.zeros((len(np_rho), 1), dtype=np.int64)
    
    for i in range(pi.shape[0]):
        r[i], p[i], d[i] = get_nb_parameters(dist.observations[i].sum)

    
    e_counts_cp = {} # count occurrences of event
    e_probs = {} # associated probabilities
    e_counts_keys = []
    for i in range(len(e)):
        k = repr(e[i])
        vec = eval(k)
        if k in e_counts_keys:
            e_counts_cp[vec] += 1
        else:
            e_counts_keys += [k]
            e_counts_cp[vec] = 1
            e_probs[vec] = mixture_probability(np.array(vec), np_rho, pi, r, p, d)
    ml = 0.
    total_counts = np.array(list(e_counts_cp.values())).sum()
    for (k,v) in e_counts_cp.items():
        ml += np.log(mixture_probability(np.array(k), np_rho, pi, r, p, d)) * v 
    BICm = ml - dist.nb_parameters * math.log(total_counts) / 2     
    Um = dist.uncertainty(data)
    ICLm = BICm - Um
    
    return(e_probs, e_counts_cp, ml, BICm, ICLm)


e_probs_m, e_counts_m, mlm, BICm, ICLm = my_icl(mixest[1].estimated, data)

# Does not work since every distribution is not a multinomial splitting negative binomial
# ICLs= [(my_icl(mixest[k].estimated, data))[3] for k in range(1,11)]

# [mixest[i].estimated.observations[k] for i in range(1,10) for k in range(i+1)]
