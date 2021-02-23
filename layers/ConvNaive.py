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
    
    # We initialize the out tensor
    H_out = int( 1 + (H + 2 * pad - FH) / stride )
    W_out = int( 1 + (W + 2 * pad - FW) / stride )
    out_shape = (N, F, H_out, W_out)
    out = np.zeros(out_shape)
    
    # We apply zero-Padding on x
    pad_2D = (pad, pad)
    pad_shape = ((0, 0), (0, 0), pad_2D, pad_2D)
    x_pad = np.pad(x, pad_shape, 'constant')
    
    # Naive Forward Convolution    
    for n in range(N):
        for f in range(F):
            for h_conv in range(H_out):
                for w_conv in range(W_out):
                    x_slice = x_pad[ n, :, h_conv*stride:h_conv*stride + FH, w_conv*stride:w_conv*stride + FW ]
                    out[n, f, h_conv, w_conv] = np.sum(x_slice * w[f]) + b[f]
    
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
    
    # We extract dimensions
    N, C, H, W = x_pad.shape
    F, _, FH, FW = w.shape
    _, _, H_out, W_out = dout.shape
    
    # We retrie the 'pad' and 'stride' parameters from conv_param
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    # We initialize the gradients
    dx_pad = np.zeros(x_pad.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    # Naive Backward Convolution    
    for n in range(N):
        for f in range(F):
            for h_conv in range(H_out):
                for w_conv in range(W_out):
                    dout_slice = dout[n, f, h_conv, w_conv]
                    # dw = convolution (x_pad, dout)
                    x_slice = x_pad[ n, :, h_conv*stride:h_conv*stride + FH, w_conv*stride:w_conv*stride + FW ]
                    dw[f] += x_slice * dout_slice
                    # dx_pad = convolution (w, dout)
                    dx_pad[ n, :, h_conv*stride:h_conv*stride + FH, w_conv*stride:w_conv*stride + FW ] += w[f] * dout_slice
                    # db
                    db[f] += dout_slice
                    
    # We remove zero padding
    dx = dx_pad[:, :, 0:(H-pad), 0:(W-pad)]
    #############################################################################
    #                             FIN DE VOTRE CODE                             #
    #############################################################################
    return dx, dw, db
