import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        # note: x_score elements w_j^T*x_i
        N = y.size(0)
        zero_to_N_indices = torch.arange(0, N) # [ 0, 1, ... N-1]
        # elements are w_{y_i}^T*x_i: each element ij is the i-th same score of real label(j)
        scores_real = x_scores[zero_to_N_indices, y].reshape(-1, 1) 
        margin_loss = self.delta + x_scores - scores_real  # DON'T FORGET - zero the i=y
        margin_loss[zero_to_N_indices, y] = 0 # zero i=y 
        margin_loss[margin_loss < 0] = 0 # max, replace negative to zero
        loss = torch.sum(margin_loss) / N # loss is (1/N)*Sigma{max(0, delta + w_j^T*x_i - w_{y_i}^T*x_i)}
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['margin_loss'] = margin_loss
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['N'] = N
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        x = self.grad_ctx['x'] # (N, D)
        M = self.grad_ctx['margin_loss'] # (N, C)
        y = self.grad_ctx['y'] #(N,)
        N = self.grad_ctx['N'] # scalar, number of samples
        # grad dimension - (D, C)
        zero_to_N_indices = torch.arange(0, N)
        M[M>0] = 1
        M[M<0] = 0
        M[zero_to_N_indices, y] = -torch.sum(M,1)
        grad = torch.mm(x.T, M)/N
        
        # ========================

        return grad
