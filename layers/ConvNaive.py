import numpy as np

def convolution_naive(x, w, b, conv_param, verbose=0):
    """
    Version naive de la propagation avant d'une convolution.  

    Le tenseur d'entrée x est une batch comprenant N images 2D de taille WxH, et 
    chacune ayant C canaux. Par exemple, si x est une batch 5 images couleur de 
    CIFAR10, alors x serait de taille 
    
    5 x 3 x 32 x 32
    
    x est donc convoluté avec F filtres dont les poids sont contenus dans "w". 
    Par exemple, si w contient 7 filtre de taille 5x5 et que les images dans x ont
    3 canaux, alors w sera de taille
    
    7 x 3 x 5 x 5
    
    Entrée:
    - x: tenseur d'entrée (N, C, H, W)
    - w: tenseur de poids du filtre (F, C, HH, WW)
    - b: vecteur de biais de taille (F)
    - conv_param: Dictionnaire comprenant les paramètres suivants:
      - 'stride': décalage horizontal et vertical lors de l'opération de convolution
      - 'pad': nombre de colonnes à gauche et à droite ainsi que de lignes en haut
               et en bas lors d'une opération de "zero padding".  Exemple, si pad = 1
               on ajoute un colonne de zéros à gauche et à droite et une ligne de
               zéros en haut et en bas.

    Retour:
    - out: tenseur convolué et taille (N, F, H', W') où H' et W' sont données par
           l'opération suivantes
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x_pad, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape

    pad = conv_param['pad']
    stride = conv_param['stride']

    #############################################################################
    # TODO: Implémentez la propagation pour la couche de convolution.           #
    # Astuces: vous pouvez utiliser la fonction np.pad pour le remplissage.     #
    #############################################################################

    # We save H' and W' values
    H_prime = int(1 + (H + 2 * pad - FH)/stride)
    W_prime = int(1 + (W + 2 * pad - FW)/stride)

    # We add the padding to x
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    # We initialize an array that will store convolutions
    out = np.zeros((N, F, H_prime, W_prime))

    # We execute the convolutions of each image with for loops
    for n in range(N):
        for i in range(H_prime):
            h_index = stride*i
            for j in range(W_prime):
                w_index = stride*j
                for f in range(F):
                    out[n, f, i, j] = (w[f] * x_pad[n:n+1, :, h_index:h_index+FH, w_index:w_index+FW]).sum() + b[f]

    # We save H' and W' in the conv_params
    conv_param['H_prime'] = H_prime
    conv_param['W_prime'] = W_prime
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    cache = (x_pad, w, b, conv_param)

    return out, cache


def backward_convolution_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.  (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x : (N, C, H, W)
    - dw: Gradient with respect to w : (F, C, HH, WW)
    - db: Gradient with respect to b : (F,)
    """
    dx, dw, db = None, None, None
    x_pad, w, b, conv_param = cache
    #############################################################################
    # TODO: Implémentez la rétropropagation pour la couche de convolution       #
    #############################################################################
    # We retrieve H', W, pad and stride from the conv params
    H_prime, W_prime = conv_param['H_prime'], conv_param['W_prime']
    pad = conv_param['pad']
    stride = conv_param['stride']

    # We extract helpful constant from data and filter shapes
    N, C, H, W = x_pad.shape
    F, C, FH, FW = w.shape

    # We initialize arrays to store gradients
    dx = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    # We calculate the backprop using for loops
    for n in range(N):
        for i in range(H_prime):
            h_index = stride*i
            for j in range(W_prime):
                w_index = stride*j
                for f in range(F):
                    dout_slice = dout[n, f, i, j]
                    dw[f] += dout_slice * x_pad[n, :, h_index:h_index+FH, w_index:w_index+FW]
                    db[f] += dout_slice
                    dx[n, :, h_index:h_index+FH, w_index:w_index+FW] += dout_slice * w[f]

    dx = dx[:, :, pad:H-pad, pad:W-pad]
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    return dx, dw, db
