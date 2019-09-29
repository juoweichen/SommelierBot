# ft_sommelier
This project is the gateway to the machine learning branch. It will help you become a master sommelier!<br>
The goal of this project is to design a single neuron perceptron machine to distinguish between high and low quality wines!

**NOTE:** We are **ONLY** allowed to import and use the matplotlib, pandas, and the standard python
libraries for this project.<br>

**NOTE:** We are **NOT** allowed to use pandas DataFrame math/matrix methods (i.e. ‘.T’, ‘.transpose’, ‘.dot’, ‘.add’, ‘.divide’, ‘.mean’, ‘.std’, etc...). neither some other mathematics libraries like: numpy, scipy, scikit- learn, tensorflow, etc.<br>

## How to start
Launch jupyter notebook<br>
For installation, check out this: [Installing Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html)<br>
open up ft_sommeliar.ipynb with jupyter notebook

## Data Description
A detailed chemical analysis has provided on a number of red and white wines (winequality-red.csv and winequality-white.csv). These analysis parameters include:
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

In order to have a better understanding on our data, we're been ask to draw a scatter matrix diagraph to show the distrubution of each parameters in pair for visual demonstration and comparison.
![image of scatter matrix](https://github.com/pootitan/ft_sommelier/blob/master/scatter_matrix.png)<br>

## Perceptron
![image of perceptron](https://github.com/pootitan/ft_sommelier/blob/master/perceptron.png)<br>
Here is a simple figure for perceptron. It receives the inputs of a sample x ,which is the feature we choose, and combines them with the weights w to compute the net input. <br>
![image of net input](https://github.com/pootitan/ft_sommelier/blob/master/netinput.png)<br>
The net input is then passed on to the activation function (here: the unit step function), which generates a binary output -1 or +1—the predicted class label of the sample. During the learning phase, this output is used to calculate the error of the prediction and update the weights.<br>
<br>
So let's implement the perceptron and feed in our wine features into it! At first we simply choose two feature which is "alclhol" & "ph". In order to illustrate the training process, we're gonna draw some plots to show what's going on in the box. <br>
![image of alcohol_ph_perceptron](https://github.com/pootitan/ft_sommelier/blob/master/alcohol_ph_perceptron.png)<br>
The plot on the left shows the error numbers and epoch of our training process. The plot on the right shows the scatter map of our wine and the decision boundary after training finished. As you can see it tooks around 13000 epochs to train our perceptron be able to distinguish the quality of wine from two features. Ok great, we made our first AI! 🎊 But unfortunately it's actually not that smart...if you mess up with features or thresholds, you will found that the training would never stop. That's because the nature of perceptron was hope to find a perfect decision boundary to classified the data to two classes, but in most of cases it's impossible, so poor perceptron stuck in training loop and could not get out. Apparently, we need something smarter then perceptron, which is the upgrade version of perceptron, ADALINE! 

## ADALINE (ADAptive LInear NEuron)
![image of ADALINE](https://github.com/pootitan/ft_sommelier/blob/master/Adaline.png)<br>

The concept of adaline is quite similiar to perceptron at a glance. The key difference between perceptron and adaline is that the weights are updated based on a linear activation function rather than a unit step function like in the perceptron. Then a quantizer, which is similar to the unit step function that we have seen before, can then be used to classify labels. <br>

One of the key ingredients of supervised machine learning algorithms is to define an objective function that is to be optimized during the learning process. This objective function is often a cost function that we want to minimize. In the case of Adaline, we can define the cost function J to learn the weights as the Sum of Squared Errors (SSE) between the calculated outcomes and the true class labels. <br>

![image of SSE](https://github.com/pootitan/ft_sommelier/blob/master/sse.png)<br>

Another nice property of this cost function is that it is convex; thus, we can use a simple, yet powerful, optimization algorithm called gradient descent to find the weights that minimize our cost function to classify
the samples.<br>

![image of alcohol_ph_adaline](https://github.com/pootitan/ft_sommelier/blob/master/alcohol_ph_adaline.png)<br>
As the result shows that Adaline was able to distinguish more complex wine quality with fair enough accuracy, and it took way less training epochs then adaline.<br>

## Standarization

## Search for good learning rate

## Training and Validation

## Multi-dimension classification

## More stuff to leaen in the future...

