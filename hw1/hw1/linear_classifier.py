import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.empty([n_features, n_classes], requires_grad=True) # empty tensor size of (n_features x n_classes)
        torch.nn.init.normal_(self.weights, mean=0, std=weight_std)             # normal distribution around 0
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        
        class_scores  = torch.mm(x, self.weights)   # S[NxC] = X[Nx(D+1)] * W[(D+1xC)]
        _, y_pred = torch.max(class_scores , 1)     # y = max element of each row (shape [N,])
        # ========================

        return y_pred, class_scores 

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = (torch.eq(y_pred, y).sum().float() / y.size(0)).item()
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            import cs236781.dataloader_utils as dataloader_utils #needed for flatten ...

            # training batches
            tot_batches = 0                                                         # need for averages
            for x_train, y_train in dl_train:
                tot_batches = tot_batches + 1
                y_pred, x_scores = self.predict(x_train)                            # predict current train batch
                total_correct += self.evaluate_accuracy(y_train, y_pred)            # add the batch acuracy
                average_loss += loss_fn.loss(x_train, y_train, x_scores, y_pred)    #batch loss
                loss_grad = loss_fn.grad()                                          # batch geadient
                loss_grad = loss_grad + torch.mul(self.weights, weight_decay)       #add regulariation
                self.weights = self.weights - loss_grad*learn_rate                  #update weights using gradient descent
            #end for batch

            # validation:
            x_valid, y_valid = dataloader_utils.flatten(dl_valid)                   # Combines batches from a DataLoader into a single tensor
            y_pred_valid, x_scores_valid = self.predict(x_valid)                    # predict validation set
            accuracy_valid = self.evaluate_accuracy(y_valid, y_pred_valid)          # validation accuracy
            valid_loss = loss_fn.loss(x_valid, y_valid, x_scores_valid, y_pred_valid) # calculate validation loss
            valid_res.loss.append(valid_loss)
            valid_res.accuracy.append(accuracy_valid)
            average_loss = average_loss / tot_batches # average by batches number (loss itself average by number of samples per batch)
            total_correct = total_correct / tot_batches # average by batches number (loss itself average by number of samples per batch)
            train_res.loss.append(average_loss)
            train_res.accuracy.append(total_correct) #
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        if has_bias:
            weights = self.weights[1:].data
        else:
            weights = self.weights.data
        size = (self.n_classes,) + img_shape # "tuple append", size Classes x C x H x W
        w_images = weights.T.reshape(size) # Work!
        ## also work:
#         w_images = torch.zeros(size) # initial to zero tensor
#         for c_class in range(self.n_classes):
#             w_images[c_class] = torch.reshape(weights[:, c_class], img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    # course staff hp:
    #     learn_rate=0.1,
    #     weight_decay=0.001,
    #     weight_std=0.001
        
#     hp['weight_std'] = .1 # X
#     hp['learn_rate'] = .001 # V
#     hp['weight_decay'] = .0001 #V
    
    
#     hp['weight_std'] = .001
#     hp['learn_rate'] = .001
#     hp['weight_decay'] = .0001
    # accuracy: 87.3%
    
    hp['weight_std'] = .003
    hp['learn_rate'] = .002
    hp['weight_decay'] = .0001
    # accuracy: 89.0%
    # ========================

    return hp
