import numpy as np
from utils.cython.im2col import col2im, im2col


class MaxPool2D:
    def __init__(self, pooling_size=3, stride=1):
        """
        Keyword Arguments:
            pooling_size {int, tuple} -- taille des filtres. (default: {3})
            stride {int, tuple} -- taille de la translation des filtres. (default: {1})

        """

        if isinstance(stride, tuple):
            self.stride = stride
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            raise Exception("Invalid stride format, must be tuple or integer")

        if isinstance(pooling_size, tuple):
            self.pooling = pooling_size
        elif isinstance(pooling_size, int):
            self.pooling = (pooling_size, pooling_size)
        else:
            raise Exception("Invalid filter format, must be tuple or integer")

        self.cache = None

    def forward(self, X, **kwargs):
        pass

    def backward(self, dA, **kwargs):
        pass

    def get_params(self):
        return {}

    def get_gradients(self):
        return {}


class MaxPool2DNaive(MaxPool2D):
    def forward(self, X, **kwargs):
        """Effectue la propagation avant d'une couche MaxPool2D

        Arguments:
            X {ndarray} -- Sortie de la couche précédente.

        Returns:
            ndarray -- Scores de la couche
        """

        N, channel, height, width = X.shape
        # TODO
        # Ajouter code ici :
        # remplacer la ligne suivante par du code de max pooling
        A = X
        
        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétro-propagation.

        Arguments:
            dA {ndarray} -- Gradients de la loss par rapport aux sorties.

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """
        # TODO
        # Ajouter code ici :
        # remplacer la ligne suivante par du code de backprop de convolution
        dX = dA


        return dX


class MaxPool2DCython(MaxPool2D):
    def forward(self, X, **kwargs):
        """Effectue la propagation avant cythonisée.

        Arguments:
            X {ndarray} -- Outputs de la couche précédente.

        Returns:
            ndarray -- Scores de la couche
        """

        N, channel, height, width = X.shape
        Pheight, Pwidth = self.pooling

        assert (height - Pheight) % self.stride[0] == 0
        assert (width - Pwidth) % self.stride[1] == 0

        out_height = np.uint32(1 + (height - Pheight) / self.stride[0])
        out_width = np.uint32(1 + (width - Pwidth) / self.stride[1])

        X_col = np.asarray(im2col(X, N, channel, height, width, Pheight, Pwidth,
                                  0, 0, self.stride[0], self.stride[1]))
        X_col = X_col.reshape(channel, Pheight * Pwidth, N, out_height * out_width)
        X_col = X_col.transpose(2, 0, 3, 1)
        X_col = X_col.reshape(N * channel * out_height * out_width, Pheight * Pwidth)

        self.cache = (X.shape, X_col)
        return np.max(X_col, axis=1).reshape(N, channel, out_height, out_width)

    def backward(self, dA, **kwargs):
        """Effectue la rétropropagation cythonisée.

        Arguments:
            dA {ndarray} -- Gradients de la loss par rapport aux sorties.

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        X_shape, X_col = self.cache
        N, channel, height, width = X_shape
        Pheight, Pwidth = self.pooling
        out_height = np.uint32(1 + (height - Pheight) / self.stride[0])
        out_width = np.uint32(1 + (width - Pwidth) / self.stride[1])

        indices = np.zeros(X_col.shape)
        indices[np.arange(X_col.shape[0]), np.argmax(X_col, axis=1)] = 1.0

        dX = indices * dA.reshape(N * channel * out_height * out_width, 1)
        dX = dX.reshape(N, channel, out_height * out_width, Pheight * Pwidth)
        dX = dX.transpose(1, 3, 0, 2)
        dX = dX.reshape(channel * Pheight * Pwidth, N * out_height * out_width)
        dX = np.asarray(col2im(dX, N, channel, height, width, Pheight, Pwidth,
                               0, 0, self.stride[0], self.stride[1]))

        return dX

    def reset(self):
        self.__init__(pooling_size=self.pooling,
                      stride=self.stride)
