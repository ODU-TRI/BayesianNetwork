# Traffic Incident Duration (TID) Prediction: An Introduction

Traffic Incident Duration (TID) refers to the time elapsed between when a traffic incident occurs and when it is resolved. This encompasses the entirety of the incident management process, from the moment of the incident's occurrence to the final resolution, including detection, response, clearance and site recovery times.

The process of TID is shown as follows [[1]](#1):
 
 <p align="center">
      <img width="50%" src="https://github.com/cogniumak/BayesianNetwork/blob/76dd2f55fcd92453f7ab49bae7190bc599244805/Figure/Timeline%20of%20TID.png"></a>
 </p>


Predicting Traffic Incident Duration accurately is of paramount importance for several reasons:

1. **Enhanced Traffic Management**: Accurate predictions of incident durations allow traffic management centers to make informed decisions about deploying resources and managing traffic disruptions.

2. **Improved Incident Response**: Predicting the duration of traffic incidents enables emergency services and first responders to better plan their operations, potentially reducing the incident's overall duration.

3. **Informed Travelers**: Travelers can be provided with better information regarding expected delays and alternative routes, thus mitigating the impact of incidents on their travel plans.

However, TID prediction is inherently challenging due to the uncertain nature of traffic incidents. Factors influencing TID include incident characteristics (type, severity, location), environmental conditions (time of day, weather), and response attributes (response time, recovery strategy).

In this context, machine learning and probabilistic graphical models such as Bayesian Networks can play a significant role in effectively modeling and predicting TID by handling uncertainty and capturing complex relationships among influencing factors.

# Traffic Incident Duration Prediction using Bayesian Networks

This README outlines how to use Bayesian Networks (BN) for predicting Traffic Incident Duration (TID).

## References

The BN model and the TID prediction approach were inspired by and built upon the ideas presented in the following work:

<a id="1">[1]</a> 
Bian, Z., & Ozbay, K. (2019). 
Final Report: Needs assessment for the development of data-driven predictive non-recurrent delay models for TRANSCOM.
Request for Access.

<a id="2">[2]</a>
Ozbay, K., & Noyan, N. (2006).
[Estimation of incident clearance times using Bayesian Networks approach](https://www.sciencedirect.com/science/article/pii/S0001457505002009?casa_token=8Jj4Vss1w-MAAAAA:9BSQ7A1TuJYxhdad2cStk5U0eNZFPRjDMxTfzIdP1rNcB2ZmImMZG3lUhFz8_VX-CvqE7evg).
*Accident Analysis & Prevention*, 38(3), 542-555. Elsevier.

<a id="3">[3]</a>
Demiroluk, S., & Ozbay, K. (2014). 
[Adaptive learning in Bayesian Networks for incident duration prediction](https://journals.sagepub.com/doi/pdf/10.3141/2460-09?casa_token=LtCbpoJrIRkAAAAA:Kqhh-oLqBAd7wdgnwx6yoIkHl97K27fVLgkBequJh0JjXNcoVJFYW_YAjvHaUHXhlLx59cBumrJcqQ). 
*Transportation Research Record*, 2460(1), 77-85. SAGE Publications Sage CA: Los Angeles, CA.

<a id="4">[4]</a>
Ozbay, K., & Kachroo, P. (1999). 
[Incident management in intelligent transportation systems](https://digitalscholarship.unlv.edu/ece_fac_articles/103/). 
*Artech House Publishers*, Norwood, MA.


## Table of Contents
1. [Introduction to Bayesian Networks](#introduction-to-bayesian-networks)
2. [Motivation for using Bayesian Networks in TID Prediction](#motivation-for-using-bayesian-networks-in-tid-prediction)
3. [Data Description](#data-description)
4. [Process of using Bayesian Networks for TID Prediction](#process-of-using-bayesian-networks-for-tid-prediction)
5. [Some quick results of using Bayesian Networks for TID Prediction](#some-quick-results-of-using-bayesian-networks-for-tid-prediction)
6. [Setup Virtual Environment and Dependencies in Jupyter Notebook](#setup-virtual-environment-and-dependencies-in-jupyter-notebook)

## Introduction to Bayesian Networks
Bayesian Networks (BN), also known as belief networks, are graphical models used to represent knowledge about an uncertain domain. It's an acyclic directed graph consisting of nodes representing variables, and edges representing dependencies between variables. Each node has a probability function that takes as input a particular set of values for the node's parent variables, and gives (as output) the probability of the variable represented by the node.

## Motivation for using Bayesian Networks in TID Prediction

Bayesian Networks (BN) are suitable for this task for several reasons:

1. **Ability to Handle Uncertainty**: Traffic incidents are inherently uncertain and dependent on a wide range of factors. BN's can manage this uncertainty effectively.

2. **Incorporation of Domain Knowledge**: BN's allow the integration of expert knowledge into the model in a transparent and interpretable way.

3. **Cause-Effect Relationships**: BN's can capture causal relationships between variables, which is critical in understanding and predicting traffic incidents.


## Data Description

The dataset used in this project includes a total of 4233 traffic incident records happend on arterials and interstate highways across New York City in the year of 2021. The main target variable is 'Duration (min)'. The target variable is converted into a categorical variable named 'Duration_class' with 4 classes, class 0: 0-30min, class 1: 30-60min, class 2: 60-90min, class 3: larger than 90min. 
|   | Duration (min) | 
|---|-----------|
| Class 0 | 0-30 |
| Class 1 | 30-60 |
| Class 2 | 60-90 |
| Class 3 | >90 |

The independent variables that influence the duration include:

1. `Direction`: Direction of the incident, 0: both directions, 1: east, 2: west, 3: south, 4: north, 5: no information.

2. `County`: The county where the incident occurred, 0: County1, 1: County2, 2: County3, 3: County4, 4: County5.

3. `Year`: The year of the incident.

4. `TOD`: Time of day (daytime or night-time) when the incident occurred, 0: daytime, 1: night-time.

5. `PeakHour`: Whether it is peak hour when the incident occurred, 0: off-peak, 1: peak-hour.

6. `DayofWeek`: The day when the incident occurred, 0: weekday, 1: weekend.

7. `SeasonofYear`: The season when the incident occurred, 0: spring and fall, 1: summer, 2: winter.

8. `Injury involved`: Whether there is injury involved in the incident, 0: no injury, 1: injuries.

9. `Truck involved`: Whether there is truck involved in the incident, 0: no heavy vehicle involved, 1: heavy vehicle involved.

10. `Lane Closure Type`: How many travel lanes closed due to the incident, 0: zero travel lane, 1: one travel lane, 2: more than two travel lanes, 3: all travel lanes.

12. `Fire involved`: Whether there is fire involved in the incident, 0: no fire, 1: fire involved.

Each record in the dataset represents a unique traffic incident.
Please note that for those incident records with large portion of missing values will be randomly assigned synthetic attributes.


## Process of using Bayesian Networks for TID Prediction

Here are the steps to use Bayesian Networks for predicting Traffic Incident Duration:

1. **Data Preprocessing**: First, preprocess the data. This may involve cleaning the data, handling missing values, and converting categorical variables into a suitable numerical format.

2. **Network Structure Learning**: Next, learn the structure of the Bayesian Network. This involves defining the nodes (variables) and edges (dependencies) of the network. 

3. **Parameter Learning**: Once the structure is defined, learn the conditional probability Distributions (CPDs) for each node in the network. 

4. **Model Validation**: Validate the model by comparing the predictions with the actual values for a subset of data. Use appropriate metrics for evaluation.

5. **Prediction**: Finally, use the learned Bayesian Network to make predictions about Traffic Incident Duration. 

## Some quick results of using Bayesian Networks for TID Prediction

Here are some quick results of using BNs for predicting TID:
1. **Learned Network Structure**: the structure learning method used in this study is [Chow-Liu ('cl')](https://ieeexplore.ieee.org/abstract/document/1054142?casa_token=Woz8snZ9j7MAAAAA:066xYu4m82tbg-ltHqDl7eXizJQNv4ImQpbkJwkAT7swJncASVHHrYsBMJCI48VCyLloFt9vPEc), the score type used in this study includes ['k2','bds','bic','bdeu'](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.3919&rep=rep1&type=pdf) for model comparison, and obtained the structure with the highest score.
The learned structure is shown as below:

 <p align="center">
      <img width="50%" src="https://github.com/cogniumak/BayesianNetwork/blob/daf80d75a19f7a16336e41d68d480e0a069fa8db/Figure/BN_structure_directed.png"></a>
 </p>
 

2. **Learned Parameter**: the estimator used for parameter learning is BayesianEstimator, the prior type used is 'bdeu'.
3. **Model Validation**: The model performance of validation data set has been evaluated using [Precision, Recall and F1-score](https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262) for each class. Here is a summary of these metrics for our model:

|   | Precision | Recall | F1-Score |
|---|-----------|--------|----------|
| Class 0 | 0.99579832 | 0.96734694 | 0.98136646 |
| Class 1 | 0.93600000 | 1.00000000 | 0.96694215 |
| Class 2 | 1.00000000 | 0.64615385 | 0.78504673 |
| Class 3 | 0.66666667 | 1.00000000 | 0.80000000 |

The array of size `[245, 117, 65, 44]` represents the support for each class, that is, the number of instances in each class in the validation set.

**Note**: The classes in the table above refer to different categories of TID defined in the 'Data Description' section. 

4. **Prediction**: the prediction performance of BNs will be evaluated using [TPR, FPR](https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262) and [AUC](https://towardsdatascience.com/roc-analysis-and-the-auc-area-under-the-curve-404803b694b9) values for each 'Duration_class':

| TID Class                      | TPR   | FPR   | AUC  |
| ------------------------------ | ----- | ----- | ---- |
| Class 0 (0-30 min)             | 0.95  | 0.01  | 0.97 |
| Class 1 (30-60 min)            | 1.00  | 0.03  | 0.98 |
| Class 2 (60-90 min)            | 0.63  | 0     | 0.81 |
| Class 3 (Larger than 90 min)   | 1.00  | 0.06  | 0.97 |
| **Overall Accuracy**           | **91.6%** | | |


*The overall accuracy score is 91.6%, which is acceptable based on [Li's paper](https://etrr.springeropen.com/articles/10.1186/s12544-018-0300-1). They summarized the acceptable accuracy of existing traffic incident duration prediction studies, ranging around 60%-70% (shown in Table 3.)
For TPR, ideally close to 1.0; >0.7 or 0.8 is often considered good. For FPR, ideally close to 0; <0.1 or 0.05 is considered good. For AUC, >0.7 is acceptable; >0.8 is good; >0.9 is excellent.

For a practical guide on how to implement these steps, refer to the following virtual environment setup and associated Jupyter notebook and codebase in this repository.

## Setup Virtual Environment and Dependencies in Jupyter Notebook
You can setup virtual environment using either requirements.txt or yml file. We show either way as follows.
### Option 1: Use requirements.txt and pip
Step 1: Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)

Step 2: Open Terminal (Windows users: anaconda prompt, macos users: terminal)

Step 3: Create Virtual Environment
```python
conda create --name bayesian-net
```

Step 4: Activate Virtual Environment
```python
conda activate bayesian-net
```

Step 5: Install Dependencies, copy paste requirements.txt to your root path. For window users, your prefix should look like C:\Users\{your device name}\. For mac users, prefix look like /Users/{your device name}/
```python
conda install pip
```

```python
pip install -r requirements.txt
```

```python
conda deactivate
```

Step 6: Add Your Created Virtual Environment to Jupyter notebook
```python
pip install --user ipykernel
```

```python
python -m ipykernel install --user --name=bayesian-net
```

Step 7: Open Jupyter Notebook and change kernel to bayesian-net

### Option 2: use bayesian-net.yml file and conda
Step 1: Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)

Step 2: Open Terminal (Windows users: anaconda prompt, macos users: terminal)

Step 3: Create Virtual Environment: copy paste bayesian-net.yml to your root path. For window users, your prefix should look like C:\Users\{your device name}\. For mac users, prefix look like /Users/{your device name}/
```python
conda env create -f bayesian-net.yml
```
```python
conda activate bayesian-net
```
Step 4: Add Your Created Virtual Environment to Jupyter notebook
```python
pip install --user ipykernel
```
```python
python -m ipykernel install --user --name=bayesian-net
```

Step 5: Open Jupyter Notebook and change kernel to bayesian-net

Happy coding and good luck with your predictions!


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file
