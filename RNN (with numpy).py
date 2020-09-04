import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

text = open('shakespeare.txt', 'r').read()

vocab = sorted(set(text))
c2i = {u:i for i, u in enumerate(vocab)}
i2c = {i:u for u, i in c2i.items()}

t2i = [c2i[i] for i in text]
ntext = np.array(t2i)

X = ntext[:-1]
Y = ntext[1:]

X_train = X[:800000]
Y_train = Y[:800000]

X_test = X[800000:]
Y_test = Y[800000:]

class INITIALIZER:

    def basic(self, hang, ryeol, scale=0.01):
        return np.random.randn(hang, ryeol) * scale

    def Xavier(self, hang, ryeol):
        return np.random.normal(size=(hang, ryeol), scale=(np.sqrt(2 / (hang + ryeol))))


class OPTIMIZER:

    def __init__(self):
        self.param = {}

    def SGD(self, dvalue, learning_rate=0.01):
        np.clip(dvalue, -5, 5, out=dvalue)

        return dvalue * learning_rate

    def adagrad(self, key, dvalue, learning_rate=0.01):
        np.clip(dvalue, -5, 5, out=dvalue)

        if key not in self.param:
            self.param.setdefault(key + "_m", np.zeros_like(dvalue))

        self.param[key + "_m"] += dvalue * dvalue

        return dvalue * learning_rate / np.sqrt(self.param[key + "_m"] + 1e-8)

class SOFTMAX:

    def __init__(self):

        pass

    def softmax(self, x):

        probs = np.exp(x - np.max(x, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)

        return probs

    def loss(self, x, y, batch_size):

        probs = self.softmax(x)
        N = x.shape[0]
        loss = - np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= batch_size

        return loss, dx


class RNN(INITIALIZER, OPTIMIZER, SOFTMAX):

    def __init__(self, vocab_size=1, h_size=1, rule="basic"):

        INITIALIZER.__init__(self)
        OPTIMIZER.__init__(self)

        self.vocab_size = vocab_size
        self.h_size = h_size

        if rule == "basic":

            self.Whh = self.basic(h_size, h_size)
            self.Wxh = self.basic(vocab_size, h_size)
            self.Why = self.basic(h_size, vocab_size)

        elif rule == "Xavier":

            self.Whh = self.Xavier(h_size, h_size)
            self.Wxh = self.Xavier(vocab_size, h_size)
            self.Why = self.Xavier(h_size, vocab_size)

        self.bh = np.zeros((1, h_size))
        self.by = np.zeros((1, vocab_size))
        self.history_loss = []

    def forward_t(self, x_t, h_t_bef):

        """
        x_t : (N, vocab_size)
        h_t_bef : (N, h_size)
        """

        h_t = np.tanh(h_t_bef.dot(self.Whh) + x_t.dot(self.Wxh) + self.bh)  # (N, h)
        y_t = h_t.dot(self.Why) + self.by  # (N, y_size)

        return y_t, h_t

    def forward(self, X, h_0=None):

        """
        X : (time, N, vocab_size)
        h_0 : (N, h_size)
        """

        time = X.shape[0]
        N = X.shape[1]

        if h_0 is None:
            h_0 = np.zeros((N, self.h_size))

        y_cache = None
        h_cache = np.array([h_0])  # (time, N, h_size)

        for x_t in X:

            y_t, h_t = self.forward_t(x_t, h_cache[-1])

            if y_cache is None:
                y_cache = np.array([y_t])
            else:
                y_cache = np.append(y_cache, [y_t], axis=0)
            h_cache = np.append(h_cache, [h_t], axis=0)

        return y_cache, h_cache

    def backward_t(self, x_t, h_t, h_bef, dy_t, dh_t):  # 글자 단위 backward

        """
        x_t : (N, vocab_size)
        h_t : (N, h_size)
        h_bef : (N, h_size)
        y_t : (N, vocab_size)
        dy_t : (N, vocab_size)
        dh_t : (N, h_size)
        """

        dWhy = h_t.T.dot(dy_t)
        dby = dy_t.sum(axis=0)
        dh_t += dy_t.dot(self.Why.T)

        dWhh = h_bef.T.dot(dh_t * (1 - h_t ** 2))
        dWxh = x_t.T.dot(dh_t * (1 - h_t ** 2))
        dh_bef = (dh_t * (1 - h_t ** 2)).dot(self.Whh.T)
        dbh = (dh_t * (1 - h_t ** 2)).sum(axis=0)

        return dWhh, dWxh, dbh, dWhy, dby, dh_bef

    def backward(self, X, dy, h_cache):

        dy = np.flip(dy, axis=0)
        h_cache = np.flip(h_cache, axis=0)

        """
        dy : (time(t ~ 1), N, y_size)
        y_cache : (time(t ~ 1), N, y_size)
        h_cahce : (time(t ~ 0), N, h_size)
        """

        dWhh = np.zeros_like(self.Whh)
        dWxh = np.zeros_like(self.Wxh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        dh_t = np.zeros((h_cache.shape[1], h_cache.shape[2]))

        for x_t in np.flip(X, axis=0):
            dWhh_t, dWxh_t, dbh_t, dWhy_t, dby_t, dh_t = self.backward_t(x_t, h_cache[0], h_cache[1], dy[0], dh_t)

            # Update
            dWhh += dWhh_t
            dWxh += dWxh_t
            dbh += dbh_t
            dWhy += dWhy_t
            dby += dby_t

            # stack pop
            dy = np.delete(dy, (0), axis=0)
            h_cache = np.delete(h_cache, (0), axis=0)

        dh_0 = dh_t

        return dWhh, dWxh, dbh, dWhy, dby

    def update(self, dWhh, dWxh, dbh, dWhy, dby, rule="SGD"):

        if rule == "SGD":

            self.Whh -= self.SGD(dWhh)
            self.Wxh -= self.SGD(dWxh)
            self.bh -= self.SGD(dbh)
            self.Why -= self.SGD(dWhy)
            self.by -= self.SGD(dby)

        elif rule == "adagrad":

            self.Whh -= self.adagrad("Whh", dWhh)
            self.Wxh -= self.adagrad("Wxh", dWxh)
            self.bh -= self.adagrad("bh", dbh)
            self.Why -= self.adagrad("Why", dWhy)
            self.by -= self.adagrad("by", dby)

    def generator(self, x, length):

        temperature = 1
        start_idx = x
        vector = np.array(start_idx).reshape(1, -1)
        gen_chars = [i2c[start_idx]]
        h = np.zeros((1, self.h_size))

        for _ in range(length):
            vector_one_hot_encoding = np.zeros((1, 1, self.vocab_size))
            vector_one_hot_encoding[0, 0, vector.reshape(-1)] = 1

            y, h = self.forward(vector_one_hot_encoding, h)  # y : (t, N, x_size)
            y = y.reshape(-1, self.vocab_size)
            probs = self.softmax(y)
            probs /= temperature
            next_char = np.random.choice(range(self.vocab_size), p=probs.reshape(-1))
            vector = np.array(next_char).reshape(-1, 1)
            h = h[-1]
            gen_chars.append(i2c[next_char])

        return ''.join(gen_chars)

    def train(self, X_train, Y_train, vocab_size=None, h_size=None, init_rule="basic", update_rule="SGD", iter=1000000,
              show=True):

        if vocab_size == None: vocab_size = len(sorted(set(X_train)))
        if h_size == None: h_size = 2 * vocab_size

        self.__init__(vocab_size, h_size, rule=init_rule)

        seq_length = min(30, max(1, X_train.shape[0] // 2))
        batch_size = min(16, X_train.shape[0])

        for i in tqdm(range(iter)):

            batch_indice = np.random.choice(X_train.shape[0] - seq_length, batch_size)
            X_batch = np.array([X_train[i:i + seq_length] for i in batch_indice])
            Y_batch = np.array([Y_train[i:i + seq_length] for i in batch_indice])

            X_batch = X_batch.T
            X_batch_one_hot_encoding = np.zeros((seq_length, batch_size, self.vocab_size))
            X_batch_one_hot_encoding[np.arange(seq_length * batch_size) // batch_size, np.arange(
                seq_length * batch_size) % batch_size, X_batch.reshape(-1)] = 1

            y_cache, h_cache = self.forward(X_batch_one_hot_encoding)

            y_cache = y_cache.reshape(seq_length * batch_size, -1)
            Real_Y = Y_batch.T.reshape(seq_length * batch_size)

            loss, dy = self.loss(y_cache, Real_Y, batch_size)
            self.history_loss.append(loss)

            dy = dy.reshape(seq_length, batch_size, -1)
            y_cache = y_cache.reshape(seq_length, batch_size, -1)

            dWhh, dWxh, dbh, dWhy, dby = self.backward(X_batch_one_hot_encoding, dy, h_cache)

            self.update(np.array(dWhh), np.array(dWxh), np.array(dbh), np.array(dWhy), np.array(dby), rule=update_rule)

            if (show == True) and (i % 10000 == 9999):
                print('epoch ', i)
                print('loss ', loss)
                print('-----------------------------------')
                gen = self.generator(np.random.randint(0, vocab_size), 200)
                print(self.generator(np.random.randint(0, vocab_size), 200))
                print('-----------------------------------')
                print()

rnn_shakespeare = RNN()

rnn_shakespeare.train(X_train, Y_train, iter = 100000, show = True)

plt.plot(rnn_shakespeare.history_loss)
plt.show()

start_idx = np.random.randint(0, 65)
rnn_shakespeare_writing = rnn_shakespeare.generator(start_idx, 1000)

print(rnn_shakespeare_writing[:1000])