import numpy as np


def cross_entropy_loss(scores, t, reg, model_params):
    """Calcule l'entropie croisée multi-classe.

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant les paramètres de chaque couche
                               du modèle. Voir la méthode "parameters()" de la
                               classe Model.

    Returns:
        loss {float} 
        dScore {ndarray}: dérivée de la loss par rapport aux scores. : Shape (N, C)
        softmax_output {ndarray} : Shape (N, C)
    """
    N = scores.shape[0]
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    softmax_output = np.zeros(scores.shape)
    
    # TODO
    # Ajouter code ici
    # We calculate a matrix with one-hot encoding as rows
    H = np.eye(C)[t]

    # We take the exponential of the scores and apply the softmax
    softmax_output = np.exp(scores)                                       # (NxC numpy array)
    softmax_output /= softmax_output.sum(axis=1).reshape((N, 1))          # (NxC numpy array)

    # We compute the loss without regularization
    B = np.log(np.sum(softmax_output*H, axis=1))                          # (Nx1 numpy array)
    loss = -1 * B.mean()

    # We add the regularization loss
    for layer in model_params.keys():
        if len(model_params[layer]) != 0:
            W, b = model_params[layer]['W'], model_params[layer]['b']
            loss += 0.5*reg*(pow(np.linalg.norm(W), 2)+pow(np.linalg.norm(b), 2))

    # We compute the gradient for the score (dL_dS * dS_dScores) = -1/S * S(H -S)
    dScores = (softmax_output-H)/N

    return loss, dScores, softmax_output


def hinge_loss(scores, t, reg, model_params):
    """Calcule la loss avec la méthode "hinge loss multi-classe".

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant l'ensemble des paramètres
                               du modèle. Obtenus avec la méthode parameters()
                               de Model.

    Returns:
        tuple -- Tuple contenant la loss et la dérivée de la loss par rapport
                 aux scores.
    """

    N = scores.shape[0]
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    score_correct_classes = np.zeros(scores.shape)

    # TODO
    # Ajouter code ici
    return loss, dScores, score_correct_classes
