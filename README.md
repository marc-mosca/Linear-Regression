<div align="center">
<pre>
██╗     ██╗███╗   ██╗███████╗ █████╗ ██████╗                                   
██║     ██║████╗  ██║██╔════╝██╔══██╗██╔══██╗                                  
██║     ██║██╔██╗ ██║█████╗  ███████║██████╔╝                                  
██║     ██║██║╚██╗██║██╔══╝  ██╔══██║██╔══██╗                                  
███████╗██║██║ ╚████║███████╗██║  ██║██║  ██║                                  
╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                                  
                                                                               
██████╗ ███████╗ ██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔════╝██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║
██████╔╝█████╗  ██║  ███╗██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║
██╔══██╗██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║
██║  ██║███████╗╚██████╔╝██║  ██║███████╗███████║███████║██║╚██████╔╝██║ ╚████║
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝ 
-------------------------------------------------------------------------------
An introduction to Machine Learning.
</pre>
</div>

The aim of this project is to introduce you to the basic concept behind machine learning.

For this project, you will have to create a program that predicts the price of a car by using a [linear function](https://en.wikipedia.org/wiki/Linear_function) train with a [gradient descent algorithm](https://en.wikipedia.org/wiki/Gradient_descent).

We will work on a precise example for the project, but once you’re done you will be able to use the algorithm with any other dataset.

## Features

You will implement a simple linear regression with a single feature - in this case, the mileage of the car.

To do so, you need to create two programs :

- The first program will be used to predict the price of a car for a given mileage. When you launch the program, it should prompt you for a mileage, and then give you back the estimated price for that mileage. The program will use the following hypothesis to predict the price :

$$
estimatePrice(mileage) = \theta_0 + (\theta_1 * mileage)
$$

- The second program will be used to train your model. It will read your dataset file and perform a linear regression on the data. Once the linear regression has completed, you will save the variables theta0 and theta1 for use in the first program. You will be using the following formulas :

$$
tmp\theta_0 = learningRate * \frac{1}{m} \displaystyle\sum_{i=0}^{m-1} (estimatePrice(mileage[i]) - price[i])
$$

$$
tmp\theta_0 = learningRate * \frac{1}{m} \displaystyle\sum_{i=0}^{m-1} (estimatePrice(mileage[i]) - price[i])
$$

Note that the estimatePrice is the same as in our first program, but here it uses your temporary, lastly computed $\theta_0$ and $\theta_1$.

Also, don’t forget to simultaneously update theta0 and theta1.

### Bonus

Here are some bonuses that could be very useful :

- Plotting the data into a graph to see their repartition.
- Plotting the line resulting from your linear regression into the same graph, to see
the result of your hard work !
- A program that calculates the precision of your algorithm.

## Usage

#### To get help with available commands:

```shell
make help
```

&nbsp;

#### First things first, you need to install dependencies:

```shell
make install
```

(or)

```shell
python3 -m pip install -r requirements.txt
```

&nbsp;

#### Now, you can run the program of your choice:

```shell
make training
```

(or)

```shell
make prediction
```

&nbsp;

#### Run the training program with bonus

```shell
make bonus
```

## Contributing

1. Fork it (<https://github.com/marc-mosca/Linear-Regression/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Contact

You can email me at mmosca@student.42lyon.fr.

<p align=center> <sub> Created while eating 🍿 by Marc MOSCA ©2024</sub></p>
