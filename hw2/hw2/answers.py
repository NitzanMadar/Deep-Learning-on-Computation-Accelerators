r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

<font color='darkgreen'>

1) The input layer has a dimension of $128*1024$, the output layer will have a dimension of $128*2048$.
We differentiate the output with respect to the input, so we get a jacobian tensor
 of a shape which is the multiplication of the dimensions - $128 \cdot 1024 \cdot 128 \cdot 2048 = 2^7 \cdot 2^{10} \cdot 2^7 \cdot 2^{11} = 2^{35}$.



2) If we use a 4 bytes (32 bits) to represent each element in our tensor, than the tensor will occupy $2^{37}$ bytes.
$1 GiB = 2^{30} Bytes$, so the tensor will occupy $ \frac{2^{37}}{2^{30}} = 2^{7} $ GiB in memory.
  
</font> 
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0.05
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.02, 0.0015, 0.00017, 0.01
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr, = 0.1, 0.00075
    # ========================
    return dict(wstd=wstd, lr=lr)

# Regarding the graphs you got for the three dropout configurations:

# 1. Explain the graphs of no-dropout vs dropout. Do they match what you expected to see?
# If yes, explain why and provide examples based on the graphs.
# If no, explain what you think the problem is and what should be modified to fix it.
# 2. Compare the low-dropout setting to the high-dropout setting and explain based on your graphs.
part2_q1 = r"""
**Your answer:**


<font color='darkgreen'>


1. [Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) is a regularization technique, that tries to overcome overfitting and reduce runtime of deep neural network [Srivastava et al 2014].
The model without the dropout (blue) overfits the data (as asked), it is characterized by the fact that the train-loss continues to decrease and the test-loss starts to increase within the training phase. 
We expect that the dropout model with good choice of `p`, will improve the original model and less overfit the test data. 
As we can see, the `dropout=0.4` graphs improves the original model - it doesn't increase the test loss and have better performance in the test accuracy. Although the reduction in the train accuracy, this model peforms better than the original.
In contrast, the `dropout=0.8` graphs shows reduction in both test and train accuracy (even the test loss is smaller), that means that `dropout=0.8` suffers from underfitting and this `p` value is too large value for dropout in our case. 

2. The low dropout setting (with `dropout=0.4`) gives us a better result and a more stable model. 
It has better accuracy and better test loss in both train and test, but it still slightly overfits, because the test loss is decaying for the first iterations and than start to increase. In contract, as mentioned before, the high dropout (`dropout=0.8`) is too high and has underfitting and lower performance.
We can see that both dropout models reduce the overfitting significantly, but has low test accuracy ($~22-30\%$). 
Hence, we can say that the dropout model improves the original overfitting model but it still does not fit the CIFAR10 database, and maybe other value of `p` or other architecture can perform better.


</font> 

"""

part2_q2 = r"""
**Your answer:**

<font color='darkgreen'>

YES. We will show a scenario where it is possibly.
We will write the cross-entropy loss function in a convenient way for this proof: 
$$\ell_{\mathrm{CE}}(\vec{y},\hat{\vec{y}}) = - {\vectr{y}} \log(\hat{\vec{y}}) = -log(\hat{y}_{True})$$
Where $\hat{y}_{True}$ indicates the prediction (probability) we got for the correct label.
Lets think of a binary classification problem (for simplicity), and imagine we have a 100 samples and we classified 90 of them correctly, where the probabily of all the correctly 
classified is 0.99 towards there correct class, and the probabilty of the ones who were classified wrong is 0.49 (towards the right class).
The loss would be proportional to $-90 \cdot log{0.99} -10 \cdot log{0.49} \approx 3.5$.
Now lets imagine in the next epoch we classify 1 more sample correctly - 91 samples correctly, therefore we increased the accuracy.
But, the probabilities are as follows: the 91 classes which were classified were classified with probability of $0.51$, and the 9 classes were classified wrong with probabilty 0.01 towards the correct class.
The loss would be proportional to $-91 \cdot log{0.51} - 9 \cdot log{0.01} \approx 44.3 $
So we gave an example where the accuracy increased as well as the loss function.


</font> 

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


<font color='darkgreen'>

1. The number of parameters is obtained by this formula:
$$ Param = FilterSize \cdot NumberOfInputChannels \cdot NumberOfFilters + NumberOfFilters $$
Where the last additive element caused by the bias term.

    We also know the following: 
$$NumberOfFilters = NumberOfOutputChannels 
\\
FilterSize = KernelSize^2$$
 * Bottleneck block:
    * 1-st step:
        * $FilterSize = 1$
        * $NumberOfInputChannels = 256$
        * $NumberOfFilters = 64$

        $\Rightarrow$ Therefore: 
        $ Params_{2} = 1 \cdot 256 \cdot 64 + 64 = 16,448 $.

    * 2-nd step:
        * $FilterSize = 9$
        * $NumberOfInputChannels = 64$
        * $NumberOfFilters = 64$
        
        $\Rightarrow$ Therefore: 
        $ Param_{2} = 9 \cdot 64 \cdot 64 + 64 = 36,928 $.

    * 3-rd step:
        * $FilterSize = 1$
        * $NumberOfInputChannels = 64$
        * $NumberOfFilters = 256$
        
        $\Rightarrow$ Therefore: 
        $ Param_{3} = 1 \cdot 64 \cdot 256 + 256 = 16,640 $.

So the total number of parameters for the Bottleneck block is:
$Params_{bottleneck} = 16,448 + 36,928 + 16,640 = 70,016 $
 * Regular block:
    * 1-st step:
        * $FilterSize = 9$
        * $NumberOfInputChannels = 256$
        * $NumberOfFilters = 256$
        
        $\Rightarrow$ Therefore: 
        $ Params_{2} = 9 \cdot 256 \cdot 256 + 256 = 590,080 $.
    * 2-nd step:
        * $FilterSize = 9$
        * $NumberOfInputChannels = 256$
        * $NumberOfFilters = 256$

        $\Rightarrow$ Therefore: 
        $ Param_{2} = 9 \cdot 256 \cdot 256 + 256 = 590,080 $.

So the total number of parameters for the Bottleneck block is:
$Params_{regular} = 590,080 + 590,080 = 1,180,160 $

---

2. We will assume the following:
* the convolution operation takes $F$ operations, where $F \equiv FilterSize \equiv f^2$ (where $FilterSize$ is the total number of element in a filter, for example, for $3 \times 3$ filter - $F=3 \cdot 3 = 9$).
* $NumberOfFilters \equiv NumberOfOutputChannels \equiv C_{out}$

* $NumberOfInputChannels \equiv C_{in}$

* Each convolution operation uses padding that takes into account the filter size ($f \times f$), so the dimension $H_{in} , W_{in}$ doesn't change along the way.

* The ReLU operation is assignment operation, we assume that this isn't count as floating point operations. In addition, this probably neglible compare to the calculation shown below.

Therefore, considering the bias operation to the output, the number of each convolution operation is given by the following formula: 
$$   \underbrace{ 
\overbrace{ C_{in}\cdot H_{in} \cdot W_{in} \cdot F \cdot C_{out}} ^\text{Multiplication} + 
\overbrace{ C_{in}\cdot H_{in} \cdot W_{in} \cdot F \cdot C_{out} - 1} ^\text{Sum}  
}_\text{Convolution} +   
\underbrace{ C_{out} \cdot  H_{in} \cdot W_{in}}_\text{Bias} =  C_{out} \cdot  H_{in} \cdot W_{in} \cdot (2 \cdot C_{in}\cdot F + 1) - 1$$

We will neglect the last element for reasons of simplicity.
In addition, after calculate the sum of all convolution operations, we need to add the operation of the shortcut sum, which is the same as the input - $C_{in} \cdot H_{in} \cdot W_{in}$
 * **Bottleneck:**
    * Main path:
        * 1-st step:
            * $F = 1$
            * $C_{in} = 256$
            * $C_{out} = 64$

            $\Rightarrow$ Therefore:
            $ Operations_{1} = 64 \cdot H_{in} \cdot W_{in} \cdot (2 \cdot 256 \cdot 1 + 1) = 32,832 \cdot H_{in} \cdot W_{in} $

        * 2-nd step:
            * $F = 9$
            * $C_{in} = 64$
            * $C_{out} = 64$

            $\Rightarrow$ Therefore: 
            $ Operations_{2} = 64 \cdot H_{in} \cdot W_{in} \cdot (2 \cdot 64 \cdot 9 + 1) = 73,792 \cdot H_{in} \cdot W_{in} $

        * 3-rd step:
            * $F = 1$
            * $C_{in} = 64$
            * $C_{out} = 256$

            $\Rightarrow$ Therefore: 
            $ Operations_{3} = 256 \cdot H_{in} \cdot W_{in} \cdot (2 \cdot 64 \cdot 1 + 1) = 33,024 \cdot H_{in} \cdot W_{in} $
            
    * Shoertcut path:
        * $ Operations_{shortcut} = 256 \cdot H_{in} \cdot W_{in} $

So the total number of operations for the Bottleneck block is:
$Params_{bottleneck} = 139,904 \cdot H_{in} \cdot W_{in} $





 * **Regular block:**
    * 1-st/2-nd step:
        * $F = 9$
        * $C_{in} = 256$
        * $C_{out} = 256$
        
        $\Rightarrow$ Therefore: 
        $ Operations_{1,2} = 256 \cdot H_{in} \cdot W_{in} \cdot (2 \cdot 256 \cdot 9 + 1) = 1,179,904 \cdot H_{in} \cdot W_{in} $
        
    * Shoertcut path:
        * $ Operations_{shortcut} = 256 \cdot H_{in} \cdot W_{in} $
        
So the total number of operations for the Regular block is:
$Params_{Regular} = 2 \cdot 1,179,904 \cdot H_{in} \cdot W_{in} + 256 \cdot H_{in} \cdot W_{in} $ = 2,360,064 \cdot H_{in} \cdot W_{in}  $

---

3. Lets talk about the characteristics of each block:
* Regular Block - makes consecutive $3 \times 3$ convolutions, therefore it combines features within feature maps.
It doesn't reduce the number of feature maps, so it doesn't combine features across feature maps.
* Residual Block - makes $1 \times 1$ convolutions, so it doesn't combine features within feature maps.
On the other hand, it reduced the number of feature maps, so in order to do that it has to combine features across feature maps to
come up with a good "compression".


</font> 

"""

part3_q2 = r"""
**Your answer:**

<font color='darkgreen'>

1. From the graphs we can see:

    * `L=2,4` has given the same accuracy more or less in both train set and test set.
    * `L=8` gave less accuracy.
    * `L=16` the network was un-trainable.
    * The result for both `K=64` and `K=32` are more or less the similar.
We think that by increasing the network's depth, the model was able to learn more complicated inputs as the filters became more specific as shown here:

<img src="https://www.researchgate.net/profile/Terje_Midtbo/publication/318967374/figure/fig3/AS:669210783002628@1536563692236/Example-of-features-that-the-filters-in-a-convolution-layer-look-for-at-different-levels.png">

But, we think that in higher `L`'s values the model was too complicate and we lost connection to the features which was input to the network. 

2. For `L=16` the network was not trainable. A reason for that could be **vanishing gradients** - by using the backpropagataion algorithm we multiply derivatives (the number of derivatives we multiply is proportional to the number of layers). If we have lots of gradients and they are less then 1, the multiplication is like a geometric series which converges to 0. The solution for that can be using a **different activation function**, which has gradients of a bigger size. 
Another solution might be **batch normalization**, which will normalize the values of each batch, and so the derivatives will be in same range in each layer (hopefully a good range).



</font> 

"""

part3_q3 = r"""
**Your answer:**


<font color='darkgreen'>


In 1.2 we changes the value `K`.
* For `L=2`: the best result was for `K=128` - and it is slightly better then what we got in experiment 1.1 for `L=2`.
* For `L=4`: the best result was for `K=256` - and it is better then what we got in experiment 1.1 for `L=4`.
* For `L=8`: the best result was for `K=32` - and it is more or less the same as what we got in experiment 1.1 for `L=8`.

The best result out of all three is for `L=4` and 'K=256', and it is better then the result we got for experiment 1.1 (better test accuracy).

We can conclude that the value of `K` is crucial in getting the best result. In our experiments, it is better to use relatively shallow net (`L=4`) with more filters per layer ('K=256') but it still suffer from overfitting (as we understood from the increase in the test loss in all of those experiment in both 1.1 and 1.2).


</font> 

"""

part3_q4 = r"""
**Your answer:**

<font color='darkgreen'>

Networks of depths `L=3,4` were un-trainable. In contrast, the shallower net (`L=1,2`) succeeded to train and got more or less the same test accuracy, but we got less accuracy than we got in the previous experiments.
So we can conclude that changing the number of features in each layer won't necessarily increase accuracy in a convolution neural net in that case.

</font> 


"""

part3_q5 = r"""
**Your answer:**


<font color='darkgreen'>

By using residual blocks, our best results are for `L=8,16` with fixed `K`, or `L=2` with changing values of `K`.
We got a slightly better result than experiment 1.3 and 1.1.
Moreover, we were able to train a network of `L=16,32`, which we weren't able in previous experiments.
So residual blocks gives us the ability to avoid vanishing gradient and check performance of a deeper network, which might be useful in some cases.


</font> 

"""

part3_q6 = r"""
**Your answer:**


<font color='darkgreen'>

1. As described earlier, our model basically based on the ResNet classifier. The changes was as the follow:
    * Convolution layer added before each residual block - we saw that the affect of using the images after using a convolusion layer improves the results, our thoughts were to feed the residual block with something that already has some manipulation (and than use it again because of the shorcut connections).
    * For each residual block:
        * Batch-normalization was added in order to avoid vanishing gradient.
        * No dropout were used, we want to keep the richness of the residual block and use this regularization at the end.
        * ReLU activation only. We read that this is a widely used activation function, and it shows great results empirically compared with other activation functions.
    * Pooling layers were max pooling only.
    * After each residual block, we added a dropout layer. We have tried to add a dropout layer within each convolution block inside the residual block, but empirically we found that using only one dropout layer at the end of each residual block gives better results in our case.
    * In case of $N mod p \neq 0$, a residual block with dropout and ReLU activation is added.



2. We got a better accuracy then all previous experiments.
The best result was obtained for `L=5` (we added that value to try to have even better test accuracy) and we got test accuracy of $80\%$ (with `L=6` we have got around $79\%$), where in experiment 1 the best test accuracy we got was about $75\%$. In addition to that, out model still suffers from overfiting (according to the test loss graph), but it is slightly better than experiment 1. We can conclude that our architecture is better in that case.


        
</font> 

"""
# ==============
