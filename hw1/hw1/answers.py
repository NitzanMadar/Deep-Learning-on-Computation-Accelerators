r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers
# Why do we need a test set? Explain:
    # 1. What does it allow us to estimate that we couldn't otherwise?
    # 2. How would it be selected from a given data set?
    # 3. How would the test-set be used at each of these points in the machine-learning pipeline: 
    #    training? cross-validation? performance evaluation? deciding whether to use one trained model or another?
part1_q1 = r"""
**Your answer:**


<font color='darkgreen'>

1. Our goal is to minimize the [generalization error](https://en.wikipedia.org/wiki/Generalization_error). 
The training part is focused on minimizing the train error.
The law of large numbers tells us, that minimizing the training error will eventually minimize the generalization error.
The model is training using the training-set, where the test-set is (typically) different. Therefore, the test-set lets us estimate the generalization error - measure of how accurately an algorithm is able to predict outcome values for previously unseen data. In other words, the test-set lets us check if our model is not [overfitting](https://en.wikipedia.org/wiki/Overfitting) the training-set.

2. Most machine-learning algorithms choose the test-set randomally using some required ratio between the trainind and the test-set. 
    * **The randomness** allow us to train a model that will not be biased due to the choice of training set. Some applications, may not any need to predict unseen data, and as a result sometime algorithms train and test on the same data. Generally, random sampler will work well. For example, if dealing with a classification problem (C classes), we would like our test set to be evenly distributed over the C classes - if that is what we expect of the real data.
It is also important to mention, that we wouldn't want the test set to be to big - because that would mean less training data.

    * **The ratio** between the training set and the test-set, is affected by the total samples in the dataset and the trained model - some models have more parameters/hyperparameters that needs to be tuned and hence a bigger training-set can be used rather then in models that have less parameters/hyperparameters - in that case we will use a bigger test-set and a smaller training-set.


3. The test-set will be used at each of these points in the machine-learning pipeline in that way:
    * **Training:** In the training point, the model uses the training-set the validation-set to update the model. Here, the test-set is not being used.
    * **Cross-validation:** As mentioned in part 2 of this homework - "for each candidate set of hyperparameters, the model is trained `K` times, each time with a different split of the training data to train and validation sets (called a fold)". So, in cross-validation the model is trained in some splits of training and validation sets, and in this phase there is no use of the test set.
    * **Performance evaluation:** The test-set is used to evaluate the model and to check how it generelarizes the problem. The model performance is evaluated on the test-set and not on the train-set, which was used in the training phase to reach high performance.
    * **Deciding whether to use one trained model or another:** According the performance, which is calculated using the test-set, we can decide if one model meets the requirements better than other models. And select the better one.For instance, if we are given 2 model that train the same data, we will choose the one that have better performance on the test set.
    
</font> 
"""


# Assuming we have already set aside some data to serve as a test set, do we always also need to split out part of the training set as a validation set? Explain your answer: If we do, why would it be wrong to use the test set instead? If we don't, why not?
part1_q2 = r"""
**Your answer:**

<font color='darkgreen'>
The validation set is used for tunning the hyperparameters of the model. 
Using the test set for tuning those hyperparameters, will cause the model to fit our test set.
This is not a good approach since the test set job is to simulate our model on unseen data, so we would have some knowledge about how good our models generalize.
Although, some models does not have hyperparameters at all, in this case we don't need any validation set (and use training and test sets only).

</font> 

"""

# ==============
# Part 2 answers
# Does increasing k lead to improved generalization for unseen data? Why or why not? Up to what point? Think about the extremal values of k.
part2_q1 = r"""
**Your answer:**  

<font color='darkgreen'>

This issue known as the *[``Bias-Variance Trade-off''](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)*

* If `k` is too small, the model relies on small number of neighbors and then it is more sensitive to noises. In other words, small `k` will have large variance and that can reduce the accuracy. 
For example, by using `k`=1, we can have a test sample that is surrounded by 3 training sample with true label but one with wrong label that is a little closer than all the others - in this case the prediction would more likely be wrong. And by using `k`=3 it would more likely be right.

    
* If `k` is  large, the low variance is offset by increased bias. In other words, large `k` means averaging more nearest neighbors for each prediction and that makes the decision boundary smoother. However, increasing `k` does not guarantee better generalization on unseen data. If k is huge then we can think of the algorithm as always picking the same prediction, which is the one which is labeled more out of all the samples.

It is easy to see that both cases are not generalizing in any way, and we need to find by practice the K that would best feet our model\data.

"""
# Question 2
# Explain why (i.e. in what sense) using k-fold CV, as detailed above, is better than:

# Training on the entire train-set with various models and selecting the best model with respect to train-set accuracy.
# Training on the entire train-set with various models and selecting the best model with respect to test-set accuracy.
part2_q2 = r"""
**Your answer:**  
<font color='darkgreen'>

1. This approach can produce hyper-parameters that overfit the train-set, and hence can have bad performance on unseen data.
Training on the entire data-set and picking the 'k'-th model according to the train-set accuracy is a bad idea, because we are validating our decision on data that we have already trained on - so obviously our validation is not true. In this way the hyperparameters are choosed without checking its accuracy on unseen data - and our main goal is that our model would give good results on unseen data.
Moreover, the fact that our model shows high performance on our train data is not an indication of a good model - it could be overfitting our data.

2. Selecting the best model with respect to test-set accuracy will use the test set as part of the training set, this is not a good choice because of the following main 2 reasons:
* Our model will be biased by the data in the test-set, since we picked our hyperparameters according to it.
* We would not know what is our true performance of our model on unseen data - so we would not know what is our approximated generalization error (no unseen data for performance evaluation).
By using K cross validation, we choose our hyperparameters by checking their performance on all the training set (each fold).
And we evaluate the "winner" on the test-set to have an approximation of the generalization error.


</font> 

"""

# ==============

# ==============
# Part 3 answers
# Question 1
# Explain why the selection of Δ>0 is arbitrary for the SVM loss 𝐿(𝑾) as it is defined above (the full in-sample loss, with the regularization term).
part3_q1 = r"""  
**Your answer:**  
<font color='darkgreen'>

The meaning of $\Delta > 0$ is the distance between the score of the true label ($s_{y_i}$) and the 
score of other (false) classes ($s_j$). As long as we have a positive margin between them, we know we classify correctly. Moreover, different values of $\Delta > 0$ will shrink\inflate $\mat{W}$, but multiplying the $\mat{W}$ weights by a constant doesn't change the plane - the classification. It is true to say that this multiplication would make the $\norm{W}$ in the Lagrangian be bigger, but by tuning $\lambda$ we can adjust to any size of $\mat{W}$. Therefore the choice of $\Delta > 0$ is arbitrary.


</font> 

"""
#  1. How do you interpret what the linear model is actually learning? Can you explain some of the classification errors based on it?
#  2. How is this interpretation similar or different from KNN?
part3_q2 = r"""
**Your answer:**  
<font color='darkgreen'>
1. The linear model predicts the data using the weights $W$, and chooses the label $j$ for the sample $i$ that has the best "response" for the dot product $x_i \cdot w_j$.
As we can see in the visualization, most of weights has shape of the number they need to predict (kind of combination of them that emphasis the strop repetitive features). Hence, by multiplying them with the sample they produce higher scores. 
To conclude we can say, that our learned weights, look some how like numbers, and if two weights look the same visually, we can expect to have some errors between the two.
For example, the first failure ("5" that predicted as "6"), we can see that the 2 weights representing "5" and "6" look kind of similar.

2. The KNN classifier is searches in the training set for the nearest `K` samples under some metric. KNN is different from linear classifiers, since the training phase in KNN is just memorizing the data. We can think of a decision boundry around the training samples, and looking in which region the test sample is.
While the learning phase in linear classifiers, is learning parameters\weights\hyperplanes - that would help us predict 
labels for unseen data.

</font> 

"""

part3_q3 = r"""
**Your answer:**  
<font color='darkgreen'>
1. The learning rate is **good**. 
* Low learning rate is characterized by slow decrease in the losses (both train and validation). In addition, in that case, the accuracy would not increase enough and the losses also will not convergence.
* High learning rate is characterized by spikes in the graphs sometimes. Moreover, by using a high learning rate we might not converge to the minimum, and the step we take might skip it (miss it). Also, the model can overfit the training set, and hence the validation loss will start to increase while the train loss not.
According to our graphs, and the characteristics mentioned above, it look like the learning rate is good.

2. According to out accuracy graph, the model is **slightly overfitting the training set**, since the validation accuracy is lower than train accuracy. 
This indicates that the model performs well on the train set and under-performs on the validation set.   
[More example og overfit / underfit loss curves](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/). In addition, continuous decrease of both losses is indicating a potential for improvement (underfit).

</font> 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

<font color='darkgreen'>
An ideal pattern in a residual plot is a straight line along the horizontal axis (where $y - \hat{y}=0$), which means that all the samples has zero residual error. In real world, there is no perfect model, and this perfect model can be an hint for overfitting, hence, the ideal pattern in real world is a residual plot of random distribution along the horizontal axis.
The linear regression model with nonlinear features after CV has better performance than the model that uses the top-5 features. It can be understood that there is nonlinear relations between the target feature and the other (that also visualized in the top 5 features vs. the target features graphs). 

</font> 

"""


# 1. Is this still a linear regression model? Why or why not?
# 2. Can we fit any non-linear function of the original features with this approach?
# 3. Imagine a linear classification model. As we saw in Part 3, the parameters  𝑾  of such a model define a hyperplane representing the decision boundary. How would adding non-linear features affect the decision boundary of such a classifier? Would it still be a hyperplane? Why or why not?
part4_q2 = r"""
**Your answer:**
<font color='darkgreen'>
1. Yes, this is still a linear model, because the linearity is in the weights, and not in the features.
For example, for the new feature $z=F(x)$, where $F$ is non-linear function, the model did not consider or know about $x$, but only the value $z$. In the original data space, this is non linear, but in the model-feature space it still a linear regression. 
2. Yes. Basically, if the model receives perfect features that simulate this non-linear function - it can be predicted by the model.
3. In the **original features space**, the decision boundary will not be hyperplane. This boundary is $ \mat{X} \mat{W} = 0$, according to that, non-linear tranformation on the features cause the decision boundary to not be an hyperplane in the original data space. Although, in the **features space**, this will remain a hyperplane (similar to the explaination in section 2 in this question).

</font> 

"""

part4_q3 = r"""
**Your answer:**


<font color='darkgreen'>

1. We searches for $\lambda$ in logarithmic scale because of the use of "strong" regularization term (squred norm, $\norm{\vec{w}}^2_2$). In other words, changing the $\lambda$ with `np.linspace`, means fixed step that that will have big effect in small number (add 0.01 to 0.001) and neglible effect in large number (add the same 0.01 to 100). But when using `np.logspace` each step will have higher influence considering the previous step, and than we can find the best value with considering the regularization temp $\norm{\vec{w}}^2_2$. In conclusion, we use a step that will have non neglible effect relatively to the previous step.

2. The model was fitted for every combination of fold, value of degree and value of lambda, that is 
$ |\lambda| \cdot |deg| \cdot |K| = 20 \cdot  3 \cdot  3 = 180 $

</font> 

"""

# ==============
