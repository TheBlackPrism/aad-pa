# aad-pa
Attacks on web-based applications and webservers are part of the everyday life in computer
science and regular security measurements are struggling to keep up with the rapidly evolving
attacks. A lot of manual work is required to keep the security systems up to date. This is not
only cumbersome but also requires very specific knowledge. As a possible solution to this
problem, conventional security systems could be complemented with machine learning.

This paper compares the utility of several anomaly detection methods, based on request logs.
By conducting a state-of-the-art analysis in the field of anomaly detection, suitable models
and feature extractions have been researched. Fine-tuning the parameters of each model
supplemented with a variety of scalers ensures, that the best possible result is achieved. The
use of two different datasets allows an accurate analysis whether the specific model is suitable
for the developed anomaly detection system or not.

The experiments have shown, that a Local Outlier Factor (LOF) model achieves the best results
regarding the correct classification, while a K-Means model stands out with its extremely low
false positive rate. However, the results also show that the effectiveness of the models in
general depends not only on the chosen features and parameters but also on the conciseness
of the used dataset.

### User Manual
This program is a console application. It can be started using any console that supports python.
The program was tested running on the following python environment:
- python 3.7.3
- numpy 1.16.4
- pandas 0.24.2
- scikit-learn 0.21.3
- sklearn 0.0
- scipy 1.2.1
- urlib3 1.24.2
- pathlib2 2.3.4

Start the application by navigating to the Code folder within your file system using the console.
Then enter:
python Anomaly_Detection.py
The application will start up and ask which dataset should be used. It will list all datasets within
the Logfiles folder. The program only needs one folder containing the following three files:
- anomalousTrafficTest
- normalTrafficTest
- normalTrafficTraining

Next up the algorithm needs to be selected. All available algorithms are listed, and only the
abbreviation needs to be entered.
Having done that, the application will ask for a scaler which can be chosen by entering one of
the abbreviations listed. If no scaler should be used this step can be skipped by pressing enter
or supplying none as selected scaler.
The program will automatically start the extraction, followed by applying the scaler and finally
applying the algorithms.

When finished the program prints the evaluation consisting of the following information:
- Dataset
- Algorithm
- Scaler
- Number of Training Samples
- Number of Anomalous Samples
- Number of Clean Samples
For each feature the performance is listed as the following:
- Trainingset Accuracy (only if algorithm supports it) -> Lists the percentage of the
trainings samples which have been identified as benign.
- True Positive -> How many of the anomalous set have been identified correctly as
anomalous.
- False Positive -> How many of the clean set have been identified incorrectly as
anomalous.

The Overall Evaluation assesses how well the algorithm performed when merging the results
of the different features together. During merging any request identified as anomalous in at
least one of the features is marked as anomalous overall.
To adjust the parameters of an algorithm the respective python file needs to be edited, before
the program is run from the console.
