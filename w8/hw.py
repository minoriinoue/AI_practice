def homework(train_X, test_X, train_y):
    import time
    train_y = np.eye(10)[train_y].astype('int32')

    train_X = train_X.reshape((train_X.shape[0], 3, 32, 32))
    test_X = test_X.reshape((test_X.shape[0], 3, 32, 32))

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1)

    def gcn(x):
        mean = np.mean(x, axis=(1,2,3), keepdims=True)
        std = np.std(x, axis=(1,2,3), keepdims=True)
        return (x - mean)/std

    class ZCAWhitening:

        def __init__(self, epsilon=1e-4):
            self.epsilon = epsilon
            self.mean = None
            self.ZCA_matrix = None

        def fit(self, x):
            x = x.reshape(x.shape[0],-1)
            self.mean = np.mean(x,axis=0)
            x -= self.mean
            cov_matrix = np.dot(x.T, x)/x.shape[0]# WRITE ME
            A, d, _ = np.linalg.svd(cov_matrix)
            self.ZCA_matrix = np.dot(np.dot(A, np.diag(1.0/np.sqrt(d+self.epsilon))), A.T)# WRITE ME

        def transform(self, x):
            shape = x.shape
            x = x.reshape(x.shape[0], -1)
            x -= self.mean
            x = np.dot(x,self.ZCA_matrix.T)
            return x.reshape(shape)
    class BatchNorm:
        #- Constructor
        def __init__(self, shape, epsilon=np.float32(1e-5)):
            self.shape = shape
            self.epsilon = epsilon

            self.gamma = theano.shared(np.ones(self.shape, dtype="float32"), name="gamma")
            self.beta = theano.shared(np.zeros(self.shape, dtype="float32"), name="beta")
            self.params = [self.gamma, self.beta]

        #- Forward Propagation
        def f_prop(self, x):
            if x.ndim == 2:
                mean = T.mean(x, axis=0, keepdims=True)# WRITE ME
                std = T.sqrt(T.var(x, axis=0, keepdims=True) + self.epsilon)# WRITE ME
            elif x.ndim == 4:
                mean = T.mean(x, axis=(0,2,3), keepdims=True)# WRITE ME (HINT : ndim=4のときはフィルタの次元でも平均をとる)
                std = T.sqrt(T.sqrt(T.var(x, axis=(0,2,3), keepdims=True) + self.epsilon))# WRITE ME

            normalized_x = (x - mean) / std # WRITE ME
            self.z = self.gamma * normalized_x + self.beta# WRITE ME
            return self.z
    class Conv:
        #- Constructor
        def __init__(self, filter_shape, function=lambda x: x, border_mode="valid", subsample=(1, 1)):

            self.function = function
            self.border_mode = border_mode
            self.subsample = subsample

            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))

            # Xavier
            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6. / (fan_in + fan_out)),
                        high=np.sqrt(6. / (fan_in + fan_out)),
                        size=filter_shape
                    ).astype("float32"), name="W")
            self.b = theano.shared(np.zeros((filter_shape[0],), dtype="float32"), name="b")
            self.params = [self.W,self.b]

        #- Forward Propagation
        def f_prop(self, x):
            conv_out = conv2d(x, self.W, border_mode=self.border_mode, subsample=self.subsample)
            self.z = self.function(conv_out + self.b[np.newaxis, :, np.newaxis, np.newaxis])
            return self.z

    class Pooling:
        #- Constructor
        def __init__(self, pool_size=(2,2), padding=(0,0), mode='max'):
            self.pool_size = pool_size
            self.mode = mode
            self.padding = padding
            self.params = []

        #- Forward Propagation
        def f_prop(self, x):
            return pool.pool_2d(input=x, ds=self.pool_size, padding=self.padding, mode=self.mode, ignore_border=True)

    class Flatten:
        #- Constructor
        def __init__(self, outdim=2):
            self.outdim = outdim
            self.params = []

        #- Forward Propagation
        def f_prop(self,x):
            return T.flatten(x, self.outdim)

    class Layer:
        #- Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function

            self.W = theano.shared(rng.normal(loc=0.0,
                        scale=np.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim,out_dim)
                    ).astype("float32"), name="W")

            self.b =  theano.shared(np.zeros(out_dim).astype("float32"), name="b")
            self.params = [ self.W, self.b ]

        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(T.dot(x, self.W) + self.b)
            return self.z

    class Activation:
        #- Constructor
        def __init__(self, function):
            self.function = function
            self.params = []

        #- Forward Propagation
        def f_prop(self, x):
            self.z = self.function(x)
            return self.z

    activation = T.nnet.relu

    layers = [                               # (チャネル数)x(縦の次元数)x(横の次元数)
        Conv((32, 3, 3, 3)),                 #   3x32x32 ->  32x30x30
        BatchNorm((32, 30, 30)),
        Activation(activation),
        Pooling((2, 2)),                     #  32x30x30 ->  32x15x15
        Conv((64, 32, 3, 3)),                #  32x15x15 ->  64x13x13
        BatchNorm((64, 13, 13)),
        Pooling((2, 2)),                     #  64x13x13 ->  64x 6x 6
        Conv((128, 64, 3, 3)),               #  64x 6x 6 -> 128x 4x 4
        BatchNorm((128, 4, 4)),
        Activation(activation),
        Pooling((2, 2)),                     # 128x 4x 4 -> 128x 2x 2
        Flatten(2),
        Layer(128*2*2, 256, activation),
        Layer(256, 10, T.nnet.softmax)
    ]

    class Optimizer(object):
        def __init__(self, params=None):
            if params is None:
                return NotImplementedError()
            self.params = params

        def updates(self, loss=None):
            if loss is None:
                return NotImplementedError()

            self.updates = OrderedDict()
            self.gparams = [T.grad(loss, param) for param in self.params]

    class AdaGrad(Optimizer):
        def __init__(self, learning_rate=np.float32(0.01), eps=np.float32(1e-6), params=None):
            super(AdaGrad, self).__init__(params=params)

            self.learning_rate = learning_rate
            self.eps = eps
            self.accugrads = [theano.shared(value=np.zeros(t.shape.eval(), dtype='float32'), name='accugrad', borrow=True) for t in self.params]

        def updates(self, loss=None):
            super(AdaGrad, self).updates(loss=loss)

            for accugrad, param, gparam in zip(self.accugrads, self.params, self.gparams):
                agrad = accugrad + gparam * gparam
                dx = - (self.learning_rate / T.sqrt(agrad + self.eps)) * gparam
                self.updates[param] = param + dx
                self.updates[accugrad] = agrad

            return self.updates

    class Adam(Optimizer):
        def __init__(self, alpha=np.float32(0.005), beta1=np.float32(0.9), beta2=np.float32(0.999), eps=np.float32(1e-8), gamma=np.float32(1-1e-8), params=None):
            super(Adam, self).__init__(params=params)

            self.alpha = alpha
            self.b1 = beta1
            self.b2 = beta2
            self.gamma = gamma
            self.t = theano.shared(np.float32(1))
            self.eps = eps

            self.ms = [theano.shared(value=np.zeros(t.shape.eval(), dtype='float32'), name='m', borrow=True) for t in self.params]
            self.vs = [theano.shared(value=np.zeros(t.shape.eval(), dtype='float32'), name='v', borrow=True) for t in self.params]

        def updates(self, loss=None):
            super(Adam, self).updates(loss=loss)
            self.b1_t = self.b1 * self.gamma ** (self.t - np.float32(1))

            for m, v, param, gparam in zip(self.ms, self.vs, self.params, self.gparams):
                _m = self.b1_t * m + (np.float32(1) - self.b1_t) * gparam
                _v = self.b2 * v + (np.float32(1) - self.b2) * gparam ** 2
                m_hat = _m / (np.float32(1) - self.b1 ** self.t)
                v_hat = _v / (np.float32(1) - self.b2 ** self.t)

                self.updates[param] = param - self.alpha*m_hat / (T.sqrt(v_hat) + self.eps)
                self.updates[m] = _m
                self.updates[v] = _v
            self.updates[self.t] = self.t + np.float32(1)

            return self.updates


    x = T.ftensor4('x')
    t = T.imatrix('t')

    params = []
    layer_out = x
    for layer in layers:
        params += layer.params
        layer_out = layer.f_prop(layer_out)

    y = layers[-1].z

    cost = T.mean(T.nnet.categorical_crossentropy(y, t))
    optimizer = Adam(params=params)

    train = theano.function(inputs=[x, t], outputs=cost, updates=optimizer.updates(cost), allow_input_downcast=True, name='train')
    valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
    test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')

    # crop
    padded = np.pad(train_X, ((0, 0),(0, 0), (4, 4), (4, 4)), mode='constant')
    crops = np.random.randint(8, size=(len(train_X), 2))
    cropped_train_X = [padded[i, :, c[0]:(c[0]+32), c[1]:(c[1]+32)] for i, c in enumerate(crops)]
    cropped_train_X = np.array(cropped_train_X)


    # Flip + crop
    flip_train_X = train_X[:, :, :, ::-1]
    flip_padded = np.pad(flip_train_X, ((0, 0),(0, 0), (4, 4), (4, 4)), mode='constant')
    cropped_flip_train_X = [flip_padded[i, :, c[0]:(c[0]+32), c[1]:(c[1]+32)] for i, c in enumerate(crops)]
    cropped_flip_train_X = np.array(cropped_train_X)


    train_X_together = np.concatenate((train_X, cropped_train_X, cropped_flip_train_X))
    print train_X_together.shape
    zca = ZCAWhitening()
    zca.fit(gcn(train_X_together))
    zca_train_X = zca.transform(gcn(train_X_together))
    print zca_train_X.shape
    train_y_together = np.concatenate((train_y, train_y, train_y))
    zca_train_y = train_y_together[:]
    print zca_train_y.shape
    zca_valid_X = zca.transform(gcn(valid_X))
    zca_valid_y = valid_y[:]

    batch_size = 100
    n_batches = zca_train_X.shape[0]//batch_size
    old = time.time()
    epoch = 0
    while (time.time() - old) < 2700.0:
        zca_train_X, zca_train_y = shuffle(zca_train_X, zca_train_y)
        for i in xrange(n_batches):
            start = i*batch_size
            end = start + batch_size
            cost = train(zca_train_X[start:end], zca_train_y[start:end])
        epoch = epoch + 1
    print 'Training cost: %.3f' % cost
    valid_cost, pred_y = valid(zca_valid_X, zca_valid_y)
    print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(zca_valid_y, axis=1).astype('int32'), pred_y, average='macro'))

    pred_y = test(test_X)
    return pred_y
