# rnn-for-text  

RNN experimentation in python

---

## Note  

For now, the codes are not mine. I am just testing out the codes available and trying to figure out how RNN works.  
Seems like, RNN isn't that hard (though it depends on the level of knowledge you already have)

---

## RNN
Recurrent Neural Networks are more advanced forms of neural networks. In RNN, the input is not limited to fixed size. But it
is sequential and temporal.  
The magical thing about RNNs are that they store the temporal values of the inputs. That is, the current state of the network 
is determined by what the input is at present as well as what the all the inputs were before.  

## Traditionally, in an ANN
In a traditional ANN, we know that:

```bash
input -> {network} -> output
```

This step is repeated with error minimization process (gradient descent) until the prediction is accurate enough.  
This means that, the network entirely depends on what **(input, output)** is supplied at present independent of what was fed before 
or what to come.  
This is a problem when we want to solve the problem where output depends on temporal values just like sequence generation. 

## Seq-to-Seq Model
One of the usecases of RNNs is the sequential generation of outputs like text generation.  
One of the traditional methods to generate sequences is **Markov Models** the output of which is pretty unrealistic.  
In comparison to markov models, the RNN is far more accurate in sequence generation.   
Although, higher markov chains can yield similar output but required more data (perhaps exponential amount?) than that required by RNN 
for similar text generation.
