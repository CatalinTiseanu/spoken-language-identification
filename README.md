# Spoken Language Identification

Winning submission for the Spoken Language Identification challenge on TopCoder (a machine learning competition platform like Kaggle).

In-depth report: https://github.com/CatalinTiseanu/spoken-language-identification/blob/master/SpokenLanguages2%20-%20report.pdf

Problem statement and dataset: https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16555&pm=13978

Results: https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16555
 (my handle is CatalinT)

## Description
* Built a language classifier which received as input a 10 second mp3 and had to classify
				 the language spoken in the mp3 - amongst 176 output language classes
* Trained a <a href="http://scikit-learn.org/stable/modules/mixture.html" target="_blank">Gaussian Mixture Model (GMM)</a>
					with 2048 components, per output language (meaning 176 GMM's with 2048 components each). GMM's had a diagonal covariance
					matrix in order to speed up the training.
* Used logistic regression as the final step to calibrate the individual GMM's prediction
* Rented 5 AWS c4.8xlarge spot instances in order to train the models in time
* Used iPython notebooks, sklearn, AWS spot machines and  <a href="https://github.com/juandavm/em4gmm">this</a> GMM library
* Client was <a href="https://www.faithcomesbyhearing.com/">Faith Comes by Hearing</a>
* The above was done within 2 weeks, in Summer 2015

## How to run

The main program is in src/main.py. To run it, use python2.7.
eg: ```python2.7 main.py init```
The necessary environment setup can be done with src/build_script.sh (for Ubuntu)
