import torch 
import torch.nn as nn
from scipy import *
import torch.optim as optim
import torch.functional as F
# import Sampling
import KraOp_Class as koc
import classical_model_method as cmm



def Train(model,seq,learn_rate=1e-1,rept_time=3):
    previous_loss = 0.
    alphabet_size = model.alphabet_size
    dim = model.dim
    min_loss = None
    for _ in range(rept_time):
        if type(model) is koc.KrausOperator:
            cur_model = koc.KrausOperator(alphabet_size, dim)
        elif type(model) is koc.cKrausOperator:
            cur_model = koc.cKrausOperator(alphabet_size, dim)
        else: 
            print("The model is not the correct type")
            break
        optimizer = optim.Adam(cur_model.parameters(), learn_rate)
        
        prev_loss = None
        for i in range(300):
            #    print(model.state_dict())
            #    model.renormalizaiton()
            loss = cur_model(seq)
            if abs(loss.item()-previous_loss) < 0.1:
                print("Current loss: {:.2f}, Previous loss {:.2f}".format(loss.item(), previous_loss))
                break

            if i % 5 == 0:
                print("Loss: {:.2f}".format(loss.item()))

            previous_loss = loss.item()

            cur_model.zero_grad()

            loss.backward()

            optimizer.step()
        if min_loss is None or min_loss > loss:
            min_loss = loss
            model = cur_model

    return model


# def Accuracy(model,target_model,M=50,L=1000):
#     # M is the number of sampled sequence.
#     # L is the sequence length
#     error = 0.
#     for j in range(M):
#         seq = cmm.EESampSeq(target_model, L)
#         error += -cmm.LogClassicalProbability(
#             target_model, seq) - model(seq).item()
#     error /= L*M

#     return -error

# This model generates the prediction accuracy between model and target model
# Input: model, target model, length of past sequence, length of future sequence
# Output: Accuracy (The KL divergence between the model and target model)


# def cond_accuracy(model, target_model, past_len=3, future_len=3):
#     alphabet_size = model.alphabet_size
#     cond_error = zeros(alphabet_size**past_len)
#     ave_cond_error = 0.
#     for i in range(alphabet_size**past_len):
#         past = decimal_to_past(i,alphabet_size, past_len)
#         log_p_past = cmm.LogClassicalProbability(target_model, past)
#         if log_p_past is NaN:
#             continue
#         for j in range(alphabet_size**future_len):
#             future = decimal_to_past(j,alphabet_size, future_len)
#             log_p_target = cmm.LogConditionalClassicalProbability(
#                 target_model, past, future)
#             if log_p_target is NaN:
#                 continue
#             diff = -model.log_cond_prob(
#                 past,future)+log_p_target
#             diff *= exp2(log_p_target)
#             cond_error[i] += diff
#         # print(cmm.LogClassicalProbability(target_model, past))
#         ave_cond_error += exp2(log_p_past)*cond_error[i]
#     return ave_cond_error/future_len


# def decimal_to_past(i, alphabet_size, past_len):
#     past = []

#     for _ in range(past_len):
#         past.append(i % alphabet_size)
#         i = i // alphabet_size
#     # past = [int(symbol) for symbol in past]
#     past.reverse()

#     return past
