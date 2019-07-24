import numpy as np
from prefixspan import PrefixSpan

def SigRank(database, min_support):
    r'''
    Compute the significance ranking of patterns in
    _database_ using minimum support min_support value
    (for PrefixSpan algorithm). Based on the following two
    papers:
	1. Gwadera, Robert, Mikhail J. Atallah, and Wojciech Szpankowski. “Reliable Detection of Episodes in Event Sequences.”
		2005. https://doi.org/10.1007/s10115-004-0174-5.
	2. Gwadera, Robert, and Fabio Crestani. “Ranking Sequential Patterns with Respect to Significance.”
		2010. https://doi.org/10.1007/978-3-642-13657-3_32.
    '''
    prefix_span = compute_prefix_span(database, min_support)

    ## Get the maximum set size and frequency distribution
    frequencies = length_frequencies(database)
    max_set_size = max(frequencies.keys())

    ## Compute alphas (mixing params)
    alphas = compute_alphas(database)

    ## Compute 0th-order model probabilities
    probs = _alphabet_probabilities(database)

    n = len(database)
    sig_rank = dict()
    for support, seq in prefix_span:
        probability = P(seq, probs, alphas, max_set_size)
        relative_support = support/n
        sig_rank[tuple(seq)] = (np.sqrt(n)*(relative_support - probability)) / np.sqrt(probability*(1-probability))

    return sig_rank

def compute_prefix_span(database, minRelSupport):
    r'''
    Accepts a list of list representing sequnces and a
    minimum support, returns the output of the
    PrefixSpan algorithm.

    Parameters
    ----------
    database: (list of lists)
        The "database" (list) of sequences.
    minRelSupport: (int)
        The minimum relative support for PrefixSpan.

    Returns
    -------
    prefix_span: (list of tuples)
        Output of PrefixSpan.frequent. List of tuples of the
        form (frequency, sequence), where sequence is a list
        representing the sequence from the database.
    '''
    ps = PrefixSpan(database)
    prefix_span = ps.frequent(minRelSupport)
    return prefix_span

def length_frequencies(database):
    r'''
    Accepts the itemset sequence database, returns a dict of the
    form <length, frequency> that stores the _frequency_ of
    sequences of length _length_.
    '''

    frequencies = dict()
    for seq in database:
        l = len(seq)
        frequencies.setdefault(l, 0)
        frequencies[l] += 1

    return frequencies

def compute_alphas(database):
    r'''
    Computes $\alpha_m$ values, where $\alpha_m$ is the
    probability of generating an itemset-sequence of size $m$.

    Parameters
    ----------
    database: (list of lists)
        The "database" (list) of sequences.

    Returns
    -------
    alphas: (np.array)
        Alpha values for each $m$ from 0 to max length.
    '''
    frequencies = length_frequencies(database)
    alphas = np.zeros(max(frequencies.keys())+1)
    n = float(sum(frequencies.values()))
    for l, frequency in frequencies.items():
        alphas[l] = frequencies[l] / n

    return alphas


def P(S, p, alphas, max_set_size):
    r'''
    Compute $P^{\exists}(S) for sequence S.
    '''
    P = 0
    for m in range(len(S), max_set_size+1):
        P += alphas[m]*P_E(S, m, p)

    assert (P>=0 and P<=1), 'P={}, not in (0,1)!'

    return P

def P_E(S, m, p):
    r'''
    Computes $P^{\exists}(w,m)$ for a sequence S.

    Parameters
    ----------
    S: (list)
        The sequence of interest
    m: (int)
        Window length
    p: (dict)
        0th-order Markov Chain probabilities for each
            element in the alphabet

    Returns
    -------
    $P^{\exists}(s|m)$: (float)
        Probability of S appearing in length _w_ sequence.
    '''
    ## NOTE: I am assuming that the probabilities that 
    ## make up p are the 0th-order Markov chain
    ## probabilities for each element in S.

    ## NOTE 2: I am reversing w and m compared to the
    ## Gwadera et al. 2005 paper to better match with
    ## the 2010 paper. Here w='m' and m=|s| as in G2010.

    seq_len = len(S)
    Q = np.zeros((m-seq_len+1, seq_len+1))

    ## Initialize P(S) NOTE: This is P_M(s) in '10
    P_S = p[S[0]]

    for j in range(seq_len):
        Q[0,j] = 1

        if j > 0:
            ## Update P(S)
            P_S = P_S*p[S[j]]
    P = 1
    for i in range(m-seq_len+1):
        Q[i,0] = (1-p[1])**i
        for j in range(1, seq_len):
            Q[i,j] = 0
            for k in range(i):
                Q[i,j] = Q[i,j] + Q[i-k, j-1]*(1-p[j])**k
        P = P + Q[i,seq_len]
    P = P_S * P

    return P

def _alphabet_probabilities(database):
    r'''
    Compute the 0th-order Markov Chain probability
    of observing each element in the alphabet.

    Parameters
    ----------
    database: (list of lists)
        The "database" (list) of sequences.

    Returns
    -------
    probs: (dict)
        Dict of the form <element, probability> for
            each element in the alphabet
    '''
    probs = dict()
    total = 0
    for seq in database:
        for e in seq:
            probs.setdefault(e, 0)
            probs[e] += 1
            total += 1
    probs = {key:freq/total for key,freq in probs.items()}

    return probs

def test_example():
    database = [
            [0, 1, 2, 3, 4],
            [1, 1, 1, 3, 4],
            [2, 1, 2, 2, 0],
            [1, 1, 1, 2, 2],
    ]

    min_support = 1

    sig_rank = SigRank(database, min_support)

    for seq, sigrank in sorted(sig_rank.items(), key=lambda kv: kv[1], reverse=True):
        print('{}: {}'.format(seq, sigrank))

if __name__ == '__main__':
    test_example()
