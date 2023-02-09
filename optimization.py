

from scipy import *
from scipy.optimize import minimize,LinearConstraint, Bounds
import numpy as np
from hmm import HMM




'''

Convert a number arbitrary base sequence.

'''

def digits(i, base = 2, pad=2):
    seq = []

    for _ in range(pad):
        seq.append(i % base )
        i = i // base
    # past = [int(symbol) for symbol in past]
    seq.reverse()

    return seq

'''
A function evaluates all morphs

'Parameters:'
--------------------
N|int the number of original states.
Na|int: the number of approximate state.

Return: 
--------------------
A list of all morphs.

'''

def all_morphs(N,Na):
    def partitions(states,Na):
        l = len(states)
        if  Na == l:
            yield [[x] for x in states]
            return
        elif Na == 1:
            yield [states]
            return
        # print(l)
        # print(states,Na)
        first = states[0]
        for smaller in partitions(states[1:],Na):
            for n,subset in enumerate(smaller):
                yield smaller[:n] + [[first]+subset] + smaller[n+1:]
        for smaller in partitions(states[1:],Na-1):
            yield  [[first]] + smaller 

    states = list(range(N))
    morphs =[]
    for pt in partitions(states,Na):
        morph = {}
        i = 0
        for subset in pt:
            if len(subset) == 1:
                continue
            morph[i] = subset
            i += 1
        morphs.append(morph)
    return morphs


'''
Generate (state, seq, prob) for a given process

'Parameters:'
--------------------
1. trans| N * D * D array: the transition probability
2. flength | int: the length of the sequences


Return: 
--------------------
(state,seq,prob)| a dict: dict[state] = (seq_number,prob) 
'''

def poss_fseq(trans, flength):
    N = len(trans)
    D,_ = trans[0].shape
    # N,D,_ = trans.shape
    ans = {}
    def prob_seq(state,seq):
        pstate = zeros(D)
        pstate[state] = 1.
        prob = 1.
        for x in seq:
            pstate = trans[x].dot(pstate)
            p = sum(pstate)
            if abs(p) <1e-10:
                return p*prob
            prob *= p
            pstate /=p
        return prob
    
    for i in range(D):
        seq_to_prob = {} 
        for j in range(N**flength):
            seq = digits(j,N,flength)
            p = prob_seq(i,seq)
            if p > 1e-10:
                # print("state={},seq={}".format(i,seq))
                seq_to_prob[j] = p
        ans[i] = seq_to_prob

    return ans

'''
Initialize the probability distribution of the approximate model's sequences

'Parameters:'
--------------------
1. morph| a list: map[merged] = (state1, state2,...)
2. state_seq_prob| a dict: state_seq_prob[state] = [(seq,prob),...]

Return: 
--------------------
(probability, nstate_to_seq)| (array,(list, list,...))

'''
def model_params(morph,state_seq_prob):
    def append_no_dup(l1,l2):
        return list((set(l1+l2)))
    nstate_to_seq = {}
    # param_length = 0

    probs = np.array([])
    for nstate, ostates in morph.items():
        allseqs = [] 
        for state in ostates:
            seq_to_prob = state_seq_prob[state]
            seqs = list(seq_to_prob)
            allseqs = sorted(append_no_dup(allseqs,seqs))
        nstate_to_seq[nstate] = allseqs

        rand_p = np.random.rand(len(allseqs))
        rand_p /= sum(rand_p)
        probs = np.append(probs,rand_p[:-1])
        # probs = probs + rand_p[:-1]   

    return (probs, nstate_to_seq) 

'''
Get the state_seq_prob from params and state_to_seq 
-------------------
Parameters
-------------------

'''

def get_state_seq_prob(params,state_to_seq):

    state_seq_prob = {}
    start = 0
    for state in state_to_seq.keys():
        seq_prob = {}
        seqs = state_to_seq[state]
        num_seq = len(seqs)
        tot_p = 0
        for i in range(num_seq-1):
            seq = seqs[i]
            seq_prob[seq] = params[start + i]

            tot_p += params[start+i] # count all the previous probability

        # add the last probability

        seq = seqs[-1] 
        seq_prob[seq] = 1- tot_p


        start += num_seq
        start -= 1
        state_seq_prob[state] = seq_prob 
    
    return state_seq_prob




'''
Evaluate kld

'Parameters:'
--------------------
1. Model parameters | 1d array: the probability distribution
2. nstate_to_seq | dict: nstate_to_seq[nstate] = seqs 
3. Morph| dict: contain the information of encoding
4. std_dtb | 1d array: the steady state distribution
5. state_seq_prob | the probability of the sequence



Return: 
--------------------
2. Distance

'''

def model_dtn(params,nstate_to_seq,morph,std_dtb,state_seq_prob):

    new_state_seq_prob = get_state_seq_prob(params,nstate_to_seq)

    # Evaluate the distance between a new state and the old state
    def state_dtn(nstate,ostate):
        dtn = 0.
        oseqprobs = state_seq_prob[ostate]
        nseqprobs = new_state_seq_prob[nstate]
        # nseqs = nstate_to_seq[nstate]


        for oseq,oprob in oseqprobs.items():
            
            # i = nseqs.index(oseq)
            nprob = nseqprobs[oseq]
            
            if abs(oprob) <1e-10:
                continue
            elif nprob <1e-10:
                return np.inf
            
            dtn += oprob*np.log2(oprob/nprob)
        return dtn
    dtn = 0.
    for nstate, ostates in morph.items():
        for ostate in ostates:
            dtn += std_dtb[ostate] * state_dtn(nstate,ostate)
    
    return dtn



'''
Distance between approximate processes and original processes.

'Parameters:'
--------------------
trans|array: the transition matrices
Na|int: the number of the approximated process's state.
flength|int: the length of future sequences

Return: 
--------------------
The minimal distance.
'''

def optimize_ekld(trans,Na,flength,opt_time=3,method = 'SLSQP'):
    N,_ = trans[0].shape 
    if N <= Na:
        return 0., None, None, None

    # trans = _hmm.trans
    state_seq_prob = poss_fseq(trans, flength)
    morphs = all_morphs(N,Na)
    # morphs = [{i:[i] for i in range(N)}] # this is for testing
    hmm_trans = HMM(trans)
    std_dtb = hmm_trans.st_state
    error = None
    for morph in morphs:
        for _ in range(opt_time):
            params,nstate_to_seq = model_params(morph,state_seq_prob) # Get params and state_to_seq

            # Set constraints
            D_params = len(params)

            # The with no parameters.
            if D_params == 0:
                return 0.,params,morph,state_seq_prob


            # Set linear constraints
            bounds = Bounds(np.zeros(D_params),np.ones(D_params))
            D_rows = len(state_seq_prob)
            con_matrix = np.zeros((D_rows,D_params)) 
            start = 0
            for i,seqs in enumerate(nstate_to_seq.values()):
                l = len(seqs)
                con_matrix[i,start:start+l-1] = 1.0
                start += l
                start -= 1
            con = LinearConstraint(con_matrix,np.zeros(D_rows),np.ones(D_rows))

            # Get the minimal result
            min_result = minimize(model_dtn, params, args=(
                            nstate_to_seq,morph,std_dtb,state_seq_prob), bounds=bounds,  constraints= con, method=method)
            
            # model_dtn(params,nstate_to_seq,morph,std_dtb,state_seq_prob)
            nerror = min_result['fun']
            if (error is None) or (error >nerror):
                error = nerror
                ans_result = min_result
                ans_morph = morph
                ans_state_to_seq = nstate_to_seq
            # print(nerror)    
    return error/flength, ans_result, ans_morph, ans_state_to_seq










