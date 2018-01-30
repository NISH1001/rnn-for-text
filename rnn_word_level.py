import numpy as np
import re

# data I/O
data = open('input.txt', 'r').read()
data = re.split(r"[ ,.\"]+", data)

words = list(set(data))
data_size, vocab_size = len(data), len(words)
print('data has {} words, {} unique.'.format(data_size, vocab_size))
word_to_ix = { ch:i for i,ch in enumerate(words) }
ix_to_word = { i:ch for i,ch in enumerate(words) }


# hyperparameters

# size of hidden layer of neurons :: n
hidden_size = 100

# number of steps to unroll the RNN for :: input length :: m
seq_length = 25
learning_rate = 1e-1

"""
    ht = activate(U*xt + W*ht-1 + b)
    yt = softmax(V*ht)
"""

# input to hidden :: U
Wxh = np.random.randn(hidden_size, vocab_size)*0.01

# hiden to hidden :: W
Whh = np.random.randn(hidden_size, hidden_size)*0.01

# hidden to output :: V
Why = np.random.randn(vocab_size, hidden_size)*0.01

# hidden bias
bh = np.zeros((hidden_size, 1))

# output bias
by = np.zeros((vocab_size, 1))

def softmax(x):
    e = np.exp(x - np.amax(x))
    return e / np.sum(e)

def train(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        # encode in 1-of-k representation
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1

        # hidden state => activate(U*xt + W*ht-a + b)
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)

        # unnormalized log probabilities for next words
        ys[t] = np.dot(Why, hs[t]) + by
        # probabilities for next words -> softmax
        ps[t] = softmax(ys[t])

        # cross entropy loss
        loss += -np.log(ps[t][targets[t], 0])

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        # backprop into y
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        # backprop into h
        dh = np.dot(Why.T, dy) + dhnext
        # backprop through tanh nonlinearity
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        # clip to mitigate exploding gradients
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed word for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)

# memory variables for Adagrad
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

# loss at iteration 0
smooth_loss = -np.log(1.0/vocab_size)*seq_length

while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        # reset RNN memory
        hprev = np.zeros((hidden_size, 1))
        # go from start of data
        p = 0
    inputs = [word_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [word_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ' '.join(ix_to_word[ix] for ix in sample_ix)
        print('----\n {} \n----'.format(txt))

    # forward seq_length wordacters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = train(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print('iter {}, loss: {}'.format(n, smooth_loss)) # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter
