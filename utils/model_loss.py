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
    
    # We compute softmax_output
    e_x = np.exp(scores - np.max(scores, axis=1, keepdims=True))   #for numerical stability
    softmax_output = e_x / e_x.sum(axis=1, keepdims=True)
    
    # We compute loss
    loss = - np.sum(np.log(softmax_output[np.arange(N), t]))
    loss /= N
    
    # We add the regularization loss
    for _, layer in model_params.items():
        params = layer.keys()
        if 'W' in params and 'b' in params:
            W, b = layer['W'], layer['b']
            loss += 0.5 * reg * (pow(np.linalg.norm(W), 2) + pow(np.linalg.norm(b), 2))
    
    dScores = softmax_output.copy()
    
    # We compute dScores 
    dScores[np.arange(N),t] -= 1
    dScores /= N

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
