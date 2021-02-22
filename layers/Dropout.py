import numpy as np


class Dropout:
    def __init__(self, drop_rate=0.2):
        """
        Keyword Arguments:
            drop_rate {float} -- pourcentage de neurones qui ne sont pas activés
                                 à l'entrainement (default: {0.2})
        """
        self.drop_rate = drop_rate

        self.cache = None

    def forward(self, X, **kwargs):
        """Application du dropout inversé lors de la propagation avant.  Voir
         
                         https://deepnotes.io/dropout 
         
         pour plus de détails.         

        Arguments:
            X {ndarray} -- Sortie de la couche précédente.

        Keyword Arguments:
            **kwargs -- Utilisé pour indiquer si le forward
                        s'applique à l'entrainement ou au test
                        et pour inclure un seed (par défaut: {'train', None})
        Returns:
            ndarray -- Sortie de la couche
        """

        mode = kwargs.get('mode', 'train')
        seed = kwargs.get('seed', None)
        A = X
        # TODO 
        # Ajouter code ici

        # We set the seed
        np.random.seed(seed)

        if mode == 'test':
            return A

        # We apply dropout depending on the mode 'train' or 'test'
        elif mode == 'train':

            # We save the probability of a neuron to not be set to 0
            p = 1 - self.drop_rate

            # We create the mask
            mask = np.random.binomial(1, p, size=A.shape) / p

            # We save the mask in the cache
            self.cache = mask

            # We apply the mask to A
            A = A * mask

            return A

        else:
            raise Exception("Invalid forward mode %s" % mode)

    def backward(self, dA, **kwargs):
        """Rétro-propagation pour la couche de dropout inversé.

        Arguments:
            dA {ndarray} -- Gradients de la loss par rapport aux sorties de la couche courante.

        Keyword Arguments:
            **kwargs -- Utilisé pour indiquer si le forward
                        s'applique à l'entrainement ou au test (default: {'train'})
        Returns:
            ndarray -- Dérivée de la loss par rapport à l'entrée de la couche.
        """

        mode = kwargs.get('mode', 'train')
        dX = dA

        # TODO
        # Ajouter code ici
        if mode == 'test':
            return dX

        elif mode == 'train':
            dX = dX*self.cache
            return dX

        else:
            raise Exception("Invalid forward mode %s" % mode)

    def get_params(self):
        return {}

    def get_gradients(self):
        return {}

    def reset(self):
        self.__init__(drop_rate=self.drop_rate)
