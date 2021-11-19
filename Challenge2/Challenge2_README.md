# HACKtheMACHINE 2021 | Track 2: Data Science Detective Bot


## Challenge 2: Optimize a Classifier

The Navy has a particular interest in deploying this capability on edge devices such as unmanned autonomous vehicles. For the second challenge, you will be developing a model that not only performs well but is also lightweight and fast giving consideration to size, weight, and power (SWAP) constraints. The emphasis will still be on F1 score of your model, but points will now also be given for inference speed and docker image and container size.

You should try to keep your docker image to a maximum of 2GB. More system/resource specific
guidance will be provided on Slack.

## SWAP (Space, Weight, and Power)

The inclusion of SWAP constraints to this challenge (Space, Weight and Power) is driven by real-world restrictions imposed by the nautical environment.  In addition to the remote distances away from physical infrastructure, some U.S. Navy platforms are very small, forcing compute resources into a restricted space, limiting memory and disk. Even an aircraft carrier, which can have multiple racks of servers, must plan for the sheer number of functions it must support: navigation, propulsion, mission planning, training, sensing, and even the Internet needs of the crew.

Limited bandwidth is another critical limitation.  Underwater communications are difficult due to the insulating nature of water, and systems afloat or aloft often rely on satellites for their comms. In times of crisis, Navy systems may find themselves in a D-DIL (Disrupted, Degraded, Intermittent, Latent) communication state.  The requirement to provide updates or patches when only high latency, kbps channels are available is a primary driver to the efficiency portion of this challenge.

## Data

The datasets are exactly the same as in Challenge 1. Again you will test your solution
by submitting your Docker image. There is no test dataset provided here.

## Performance Metrics

Performance metrics are used to evaluate your machine learning algorithm and pipeline. Metrics provide insight into how your model is performing under the hood.

| Metric | Definition | Formula |
|--------|------------|---------|
|Precision| Fraction of relevant instances among the retrieved instances| Precision = TP / (TP + FP) |
|Recall (Sensitivity) | Fraction of relevant instances that were retrieved | Recall = TP / (TP + FN) |
|F1-Score | Measure of a tests accuracy, the harmonic mean of the precision and recall | F1-Score = (2 x Precision x Recall ) / (Precision + Recall ) |


Sources: https://en.wikipedia.org/wiki/Precision_and_recall, https://en.wikipedia.org/wiki/F-score


### Python Tutorials:
* [Timeit Examples](https://www.geeksforgeeks.org/timeit-python-examples/) <br>
* [Timeit Examples 2](https://www.guru99.com/timeit-python-examples.html)
* [Python Time Module](https://realpython.com/python-time-module/) <br>
* [Metrics to Evaluate your ML Algorithm](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)<br>
* [Evauluate ML Performance Using Resampling](https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/) <br>
* [How Much Memory is your Machine Learning Code Consuming?](https://www.kdnuggets.com/2021/07/memory-machine-learning-code-consuming.html) <br>
* [Monitoring Memory Usage of a Running Python Program](https://www.geeksforgeeks.org/monitoring-memory-usage-of-a-running-python-program/)

### Python Modules:
* [Time](https://docs.python.org/3/library/time.html) - Time access and conversions
* [Timeit](https://docs.python.org/3/library/timeit.html) - Measure execution time of small code snippets
* [Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) - Classification metrics



## Challenge 2 Submission Instructions

For Challenge #2, you can continue your work on your initial submission, or start a new approach.  Either way, you will need to create a model that works well, and is efficient in terms of size and speed.


Challenge 2 submission has two parts (same as Challenge 1):
1) Any code used to train a classifier or perform analysis should be checked into the appropriate folder in your designated GitLab repo (i.e. Challenge 2 code in the `Challenge 2` folder).
2) A docker image that contains your model and inference code will be pushed to your designated image registry.

## Challenge 2 Submission Deadline

Please submit your final work for Challenge 2 by 1700 EST on Thursday, 18 November.

## Challenge 2 Grading

Much like challenge 1, this challenge will have a leaderboard where scores will be provided as submissions are analyzed and graded.

**Challenge 2 will be worth a total of 60 points.**

| Criteria | Description | Criteria % Weight |
|----------|-------------|-------------------|
|**Model Performance & Efficiency** | A function of model performance, inference speed, and docker image size.  | 95 |
|**Documentation** | Any code used to train models, perform analysis in git with reasonable levels of commenting. |5 |
