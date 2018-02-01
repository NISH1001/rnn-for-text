#!/usr/bin/env python3

import numpy as np
from dataloader import load_data
from featurizer import process_text, tokens

def softmax(x):
    e = np.exp(x - np.amax(x))
    return e / np.sum(e)

class RNN:
    """
        zt = U*xt + W*ht-1 + b
        ht = tanh(zt)
        yt = V*ht
        pt = softmax(yt)
        Jt = cross_entropy(targets, pt)
    """
    def __init__(self, hidden_size, vocab_size, learning_rate):
        # U
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        # V
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        # W
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01

        self.hprev = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def train(self, int_text, iteration=10, seq_length=16):
        """
            int_text is word-embeddded whole data
                Text with each word as int
        """
        smooth_loss = -np.log(1.0/self.vocab_size) * seq_length
        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        # memory variables for Adagrad
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)

        for i in range(iteration):
            # we're sweeping from left to right in steps seq_length long
            if p+seq_length+1 >= len(int_text) or i == 0:
                # reset RNN memory
                self.hprev = np.zeros((self.hidden_size, 1))
                # go from start of data
                p = 0

            inputs = int_text[p : p+seq_length]
            targets = int_text[p+1 : p+seq_length+1]

            loss_iter, xs, hs, ys, ps = self.feed_forward(inputs, targets)
            smooth_loss = smooth_loss * 0.999 + loss_iter * 0.001
            if i % 100 == 0:
                print('iter {}, loss: {}'.format(i, smooth_loss))

            dWxh, dWhh, dWhy, dbh, dby, hprev = self.backpropagate(inputs, targets, xs, hs, ys, ps)
            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                        [dWxh, dWhh, dWhy, dbh, dby],
                                        [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                # adagrad update
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

    def feed_forward(self, inputs, targets):
        """
            int_text,targets are both list of integers.
            returns the loss, and other states
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.hprev)
        loss = 0

        for t in range(len(inputs)):
            # encode in 1-of-k representation
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1

            # hidden state => activate(U*xt + W*ht-a + b)
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

            # probabilities for next words -> softmax
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = softmax(ys[t])

            # cross entropy loss
            #loss += -np.log(ps[t][targets[t], 0])
        return loss, xs, hs, ys, ps

    def backpropagate(self, inputs, targets, xs, hs, ys, ps):
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])

            # backprop into y
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            # backprop into h
            dh = np.dot(self.Why.T, dy) + dhnext

            # backprop through tanh nonlinearity
            # der(tanh) => (1 - (tanh)^2)
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            # clip to mitigate exploding gradients
            np.clip(dparam, -5, 5, out=dparam)
        return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(self, seed_ix, n):
        """
            sample a sequence of integers from the model
            h is memory state, seed_ix is seed word for first time step
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.hprev) + self.bh)
            y = np.dot(self.Why, self.hprev) + self.by
            p = softmax(y)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

def main():
    #data = "hello i am paradox. i am nishan. i am gru"
    data = load_data("data/input.txt")

    hidden_size = 100
    seq_length = 25
    learning_rate = 1e-1
    int_text, vocab_to_int, int_to_vocab = process_text(data)
    vocab_size = len(vocab_to_int)

    rnn = RNN(hidden_size, vocab_size, learning_rate)
    rnn.train(np.array(int_text), 10000, seq_length)

    sample_idx = rnn.sample(int_text[0], 20)
    generated_text = ' '.join(int_to_vocab[idx] for idx in sample_idx)

    for key, token in tokens.items():
        generated_text = generated_text.replace(' ' + token.lower(), key)
        generated_text = generated_text.replace('\n ', '\n')
        generated_text = generated_text.replace('( ', '(')
    print(generated_text)

if __name__ == "__main__":
    main()

