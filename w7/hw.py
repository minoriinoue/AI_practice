class Conv:
    #- Constructor
    def __init__(self, filter_shape, function, border_mode="valid", subsample=(1, 1)):
        # filter shape (k, l, i, j): 4次元
        #  - k: フィルタ数(出力のチャネル数)
        #  - l: 入力のチャネル数
        #  - i: フィルタの行数
        #  - j: フィルタの列数

        self.function = function
        self.border_mode = border_mode
        self.subsample = subsample

        self.W = theano.shared(rng.uniform(low=-0.08, high=0.08, size=filter_shape).astype('float32'))
        self.b = theano.shared(np.zeros(filter_shape[0]).astype('float32'))

        self.params = [self.W, self.b]

    #- Forward Propagation
    def f_prop(self, x):
        conv_out = conv2d(x, self.W, border_mode=self.border_mode, subsample=self.subsample)# WRITE ME! (HINT: conv2dを使って畳み込み処理をする)
        self.z   = self.function(conv_out + self.b[np.newaxis,:,np.newaxis, np.newaxis])# self.b.reshape(1, filter_shape, 1, 1)WRITE ME! (HINT: バイアスを加えて活性化関数をかける. bの次元に注意)
        return self.z

class Pooling:
    #- Constructor
    def __init__(self, pool_size=(2, 2), mode='max'):
        self.pool_size = pool_size
        self.mode = mode
        self.params = []

    #- Forward Propagation
    def f_prop(self, x):
        return pool.pool_2d(input=x, ds=(2, 2), mode='max', ignore_border=True)# WRITE ME! (HINT: pool.pool_2dを使ってプーリングを行う)

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

        self.W = theano.shared(rng.uniform(
                    low=-np.sqrt(6. / (in_dim + out_dim)),
                    high=np.sqrt(6. / (in_dim + out_dim)),
                    size=(in_dim,out_dim)
                ).astype("float32"), name="W")
        self.b =  theano.shared(np.zeros(out_dim).astype("float32"), name="b")
        self.params = [ self.W, self.b ]

    #- Forward Propagation
    def f_prop(self, x):
        self.z = self.function(T.dot(x, self.W) + self.b)
        return self.z

#--- Stochastic Gradient Descent
def sgd(params, g_params, eps=np.float32(0.1)):
    updates = OrderedDict()
    for param, g_param in zip(params, g_params):
        updates[param] = param - eps * g_param
    return updates

activation = T.tanh

layers = [
    Conv((20, 1, 5, 5),activation),  # 28x28x 1 -> 24x24x20
    Pooling((2, 2)),                 # 24x24x20 -> 12x12x20
    Conv((50, 20, 5, 5),activation), # 12x12x20 ->  8x 8x50
    Pooling((2, 2)),                 #  8x 8x50 ->  4x 4x50
    Flatten(2),
    Layer(4*4*50, 500, activation),
    Layer(500, 10, T.nnet.softmax)
]

x = T.ftensor4('x')
t = T.imatrix('t')

params = []
layer_out = x
for layer in layers:
    params += layer.params
    layer_out = layer.f_prop(layer_out)

y = layers[-1].z

cost = T.mean(T.nnet.categorical_crossentropy(y, t))

g_params = T.grad(cost, params)
updates = sgd(params, g_params)

train = theano.function(inputs=[x, t], outputs=cost, updates=updates, allow_input_downcast=True, name='train')
valid = theano.function(inputs=[x, t], outputs=[cost, T.argmax(y, axis=1)], allow_input_downcast=True, name='valid')
test  = theano.function(inputs=[x], outputs=T.argmax(y, axis=1), name='test')

batch_size = 20
n_batches = train_X.shape[0]//batch_size
for epoch in xrange(5):
    train_X, train_y = shuffle(train_X, train_y)
    for i in xrange(n_batches):
        start = i*batch_size
        end = start + batch_size
        train(train_X[start:end], train_y[start:end])
    valid_cost, pred_y = valid(valid_X, valid_y)
    print 'EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, axis=1).astype('int32'), pred_y, average='macro'))
