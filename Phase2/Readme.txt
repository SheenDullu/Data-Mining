		------DETAILS------
Requirements
	Python version3.7

Execution
	To train the models run the below command
		python Train.py
	To test the models run the below command
		python Test.py <Test File Name>  
		For Example -> python Test.py test_data.csv]

Output Interpretation
	After the test command runs, 4 seperate files containing test data predictions for each of the model are generated.
	File name format - <Model Name>_output.csv [Examples -> Gradient_Boosting_output.csv, Random_Forest_output.csv, 
	ML_Perceptron_output.csv, SVM_RBF_output.csv]


		------SUBMISSION DETAILS------
Readme

Train.py
	Contains code for feature generation and training all the models

Test.py
	Contain code for feature generation and testing all the models

models (folder)
	Contains the saved model objects in pickle format


		------TESTING RESULTS------
-----Random_Forest-----
Accuracy: 0.6665397536394175 F1: 0.6605804265775191
Recall: 0.6386904761904761 Precision: 0.6851411792342026
-----Gradient_Boosting-----
Accuracy: 0.6919148936170212 F1: 0.691706681009617
Recall: 0.6801870748299319 Precision: 0.7048571428571428
-----SVM_RBF-----
Accuracy: 0.700515117581187 F1: 0.6829485396746546
Recall: 0.6385204081632653 Precision: 0.7352441710336448
-----ML_Perceptron-----
Accuracy: 0.711063829787234 F1: 0.7060395527960815
Recall: 0.6924319727891157 Precision: 0.7240131691610955


		------IMPLEMENTATION DETAILS------
Features used 
	Sliding window statistics like mean, skew, AUC, standard deviation 
	CGM Velocity window statstics such as mean, std, skew, max, min
	CGM Velocity zero crossing, the magnitude of the slope change at zero crossing
	Top 5 FFT values
	Polynomial coefficents 

For normalization
	Before passing to the classifiers, data is scaled to zero mean and unit standard deviation(StandardScaler) 

Models
	Gradient Boosting/XGBoost
	Multi Layer Perceptron (Feed forward neural network)
	Random Forest
	SVM (RBF kernel)

For Cross Validation
	I performed Stratified 5 fold cross validation.





