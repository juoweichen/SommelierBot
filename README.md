# ft_sommelier
This project is the gateway to the machine learning branch. It will help you become a master sommelier!<br>
The goal of this project is to design a single neuron perceptron machine to distinguish between high and low-quality wines!
 
**NOTE:** We are **ONLY** allowed to import and use the matplotlib, pandas, and the standard python
libraries for this project.<br>
 
**NOTE:** We are **NOT** allowed to use pandas DataFrame math/matrix methods (i.e. ‘.T’, ‘.transpose’, ‘.dot’, ‘.add’, ‘.divide’, ‘.mean’, ‘.std’, etc...). neither some other mathematics libraries like numpy, scipy, scikit- learn, tensorflow, etc.<br>
 
## How to start
Launch jupyter notebook<br>
For installation, check out this: [Installing Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html)<br>
open up ft_sommeliar.ipynb with jupyter notebook
 
## Data Description
A detailed chemical analysis has provided a number of red and white wines (winequality-red.csv and winequality-white.csv). These analysis parameters include:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Beside above parameters, there's a “Quality” parameter which is a simulation of the score that three trained wine tasters would give the wine. Thus, each row in the dataset describes the chemical characteristics and quality score for a particular wine.
 
In order to have a better understanding of our data, we're been ask to draw a scatter matrix diagraph to show the distribution of each parameter in pairs for visual demonstration and comparison.
![image of scatter matrix](https://github.com/pootitan/ft_sommelier/blob/master/img/img/scatter_matrix.png)<br>
 
## Perceptron
![image of perceptron](https://github.com/pootitan/ft_sommelier/blob/master/img/perceptron.png)<br>
Here is a simple figure for the perceptron. It receives the inputs of sample x, which is the feature we choose and combines them with the weights w to compute the net input. <br>
![image of net input](https://github.com/pootitan/ft_sommelier/blob/master/img/netinput.png)<br>
The net input is then passed on to the activation function (here: the unit step function), which generates a binary output -1 or +1—the predicted class label of the sample. During the learning phase, this output is used to calculate the error of the prediction and update the weights.<br>
<br>
So let's implement the perceptron and feed in our wine features into it! At first, we simply choose two feature which is "alcohol" & "ph". In order to illustrate the training process, we're gonna draw some plots to show what's going on in the box. <br>
![image of alcohol_ph_perceptron](https://github.com/pootitan/ft_sommelier/blob/master/img/alcohol_ph_perceptron.png)<br>
The plot on the left shows the error numbers and epoch of our training process. The plot on the right shows the scatter map of our wine and the decision boundary after training finished. As you can see it took around 13000 epochs to train our perception to be able to distinguish the quality of wine from two features. Ok great, we made our first AI! 🎊 But unfortunately it's actually not that smart...if you mess up with features or thresholds, you will found that the training would never stop. That's because the nature of the perceptron was hope to find a perfect decision boundary to classified the data into two classes, but in most cases it's impossible, so poor perceptron stuck in the training loop and could not get out. Apparently, we need something smarter then perceptron, which is the upgrade version of perceptron, ADALINE!
 
## ADALINE (ADAptive LInear NEuron)
![image of ADALINE](https://github.com/pootitan/ft_sommelier/blob/master/img/Adaline.png)<br>
The concept of Adaline is quite similar to the perceptron at a glance. The key difference between perceptron and Adaline is that the weights are updated based on a linear activation function rather than a unit step function like in the perceptron. Then a quantizer, which is similar to the unit step function that we have seen before, can then be used to classify labels. <br>
 
One of the key ingredients of supervised machine learning algorithms is to define an objective function that is to be optimized during the learning process. This objective function is often a cost function that we want to minimize. In the case of Adaline, we can define the cost function J to learn the weights as the Sum of Squared Errors (SSE) between the calculated outcomes and the true class labels. <br>
 
![image of SSE](https://github.com/pootitan/ft_sommelier/blob/master/img/sse.png)<br>
Another nice property of this cost function is that it is convex; thus, we can use a simple, yet powerful, optimization algorithm called gradient descent to find the weights that minimize our cost function to classify the samples.<br>
 
![image of alcohol_ph_adaline](https://github.com/pootitan/ft_sommelier/blob/master/img/alcohol_ph_adaline.png)<br>
As the result shows that Adaline was able to distinguish more complex wine quality with fair enough accuracy, and it took way less training epochs then Adaline.<br>
 
![image of loss function](https://github.com/pootitan/ft_sommelier/blob/master/img/loss_func.png)<br>
The above plot shows that the cost of each epoch. The cost droped dramatically at the begining and become gradually soon because gradient getting flatten when it closer to the convex, which is the minimum possible cost we can get.<br>
 
## Search for good learning rate
![image of loss function](https://github.com/pootitan/ft_sommelier/blob/master/img/loss_func.png)<br>
Looking for a good learning rate is essential in gradient descending algorithms because as the figure on the right shows that, if the learning rate is too large, it would fall to converge because net input overshooting the global minimum. On the other hand, if the learning rate is too small, it might be descending slower so it would need more epoch to reach the global minimum.<br>
 
In order to find the best possible learning rate, I clamped the range of learning rates by experimenting with rounds of training. If the result diverge then I lower the learning rate upper limit, on the other hand, if the result took too much epoch to train, I increase the bottom limit.<br>
 
## Training and Validation
We overcome a lots of troubles in order to reach our best possible classified result, and we finally got it! Hooray!<br>
But the trues is, the result probably gonna kill in lab but adopted poorly to real-world data, why is this happened? To explain this we need to understand overfitting. <br>
![image of overfitting](https://github.com/pootitan/ft_sommelier/blob/master/img/overfitting.png)<br>
Overfitting is a common problem in machine learning, where a model performs well on training data but does not generalize well to unseen data (test data). If a model suffers from overfitting, we also say that the model has a high variance, which can be caused by having too many parameters that lead to a model that is too complex given the underlying data. Similarly, our model can also suffer from underfitting (high bias), which means that our model is not complex enough to capture the pattern in the training data well and therefore also suffers from low performance
on unseen data.<br>
 
There's several way to fight the overfitting, one of them is using k-fold cross-validation. We split the data to k equal size set, and in each round, we choose a set of data as the validation set and feeding the rest of the data to train, after every sub-set had been chosen, the validation ends. <br>
![image of k-fold](https://github.com/pootitan/ft_sommelier/blob/master/img/kfold.png)<br>
The k-fold result shows that, after 5 rounds of validation, the average accuracy was 0.84, not too far from our target accuracy 0.85.
 
## Multi-dimension classification
Finally we had a simple machine learning process, includes data rendering, perceptron, Adaline, validation, etc. From here we keep on feeding more dimension to see if our model can handling more complex datasets. There are whole lots of hyperparameters we can tuning, like features, learning rate, data threhold, k-fold, etc. Try it on Jupyter notebook!
 
## More stuff to leaen in the future...
We had build two models base on percptron and adaline, those two algorithms were been designed to be a linear classified model. In reality, we rarely seem linear models. Instead, most of the data were non-linear pattern. One of the approach to achieved non-linear classification is add more nuron and layer, and that's the field of deep learning. Hope this project would give us some guidelines on the path to AI. Cheers!
 

