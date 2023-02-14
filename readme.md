## Introduction

The code in this package serves as the primary implementation of the research paper titled '*Provable Superior Accuracy in Machine Learned Quantum Models*' (see [arxiv](https://arxiv.org/abs/2105.14434)).

## Usage 

There are two main functions 

1. Train a model from sequential data. 

```python

seq = data_seq
model = KraOp_Class.KrausOperator(alphabet_size,dim)
model = Train.Train(model,seq,rept_time=N)
```

2. Get a lower bound on the distortion of the optimal approximate model with a given dimension 

```python
optimization.optimize_ekld(transition_matrix,given_dimension,flength,opt_time=N)[0]
```

For more examples, see `examples.ipynb`

## License 

This package is licensed under the MIT License.