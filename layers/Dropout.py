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
        # Appliquer un dropout en fonction du mode 'train' ou 'test'
        # "seed" sert à initialiser np.random
        # ne pas oublier de mettre dans la self.cache le masque de dropout.
        
        #A = np.copy(X)
        
        mask = None
        keep_rate = 1 - self.drop_rate
        
        if seed != None:
            np.random.seed(seed)
        
        if mode == 'train':
            mask = ( np.random.rand(*A.shape) < keep_rate ) / keep_rate
            A = (A * mask)
        
        elif mode == 'test':
            pass
        
        self.cache = mask
        
        return A

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
        # Rétro-propagation du gradient en fonction du mode 'train' ou 'test'
        # n'oubliez pas que le "masque" de dropout est dans la cache!
        
        mask = self.cache
        keep_rate = 1 - self.drop_rate
        
        if mode == 'train':
            
            dX = (dA * mask) / keep_rate
        
        elif mode == 'test':
            pass

        return dX

    def get_params(self):
        return {}
        
    def get_gradients(self):
        return {}

    def reset(self):
        self.__init__(drop_rate=self.drop_rate)
