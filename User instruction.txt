////////////////////////////////////////////////////////////////////////////////////////////
\                              SMARTINvest                                                  \
////////////////////////////////////////////////////////////////////////////////////////////



 	Description
        -----------
 This folder contains 2 scripts:

-----An R Script for data cleaning and monte carlo simulation of portfolios to train the machine learning classifer
-----A python script  to classify portfolios into 2 classes (good & poor performance) using convolutional neural networks.


----------------------------------data cleaning and monte carlo simulation of portfolios to train the machine learning classifer----------------------------------------
 Table of Contents
 ~~~~~~~~~~~~~~~~~~
 
 1. Content
 2. Getting Started
 3. Usage
 
 	1.Included	
         ----------
 * within this folder you can find a subfolder entitled 'Data cleaning & Preparation' in which you can find the following:
 - R workspace: contains every dataset required for the portoflio simulation and potential applications.
 - Text file 'R packages installer_codes'
 - Text file 'Import quote history'
 - Text file 'Import key metrics'
 - Text file 'R workspace code'
 - Text file 'Monte carlo simulation to train the classifier_codes'

 	2.Getting Started
         ----------------
In order to work with these scripts and workplace you will need the latest version of R accompanied with R studio.
Install:
 - R: https://cran.r-project.org/bin/windows/base/
 - R studio: https://www.rstudio.com/products/rstudio/download/

To install packages openup R studio IDE and run the 'R packages installer_codes' 

 	2.Usage
         ------
In order to generated the simulated portoflios you have:
 - Import the R workspace
 - Run Text file 'Monte carlo simulation to train the classifier_codes' and remember to change the last 2 code lines to save the data to a folder of your choice
 - the data can then be used as input for the classifier 


In order to generate a working space with all the relevant data you will have to:
 - Gain access to thomson Reuters terminal and download the quote history and keymetrics of a selected set of stocks
 - Key metrics & Quote history should be downloaded to seperate folders
 - Run the Text file 'Import quote history' after setting the right working directory 
 - Run the Text file 'Import key metrics' after setting the right working directory 
 - Run the  - Text file 'R workspace code' to create the final workspace. 
******Note you might face some troubles given that the script is designed for a specific working enviroment for which we advise you to load the workspace******


----------------------------------script  to classify portfolios into 2 classes (good & poor performance) using convolutional neural networks---------------------------


 Table of Contents
 ~~~~~~~~~~~~~~~~~~
 
 1. Content
 2. Getting Started
 3. Usage
 4. Results
 
 	1.Included	
         ----------
 * This folder content:
 - A python file "final_code.py"
 - A folder "data" that contains the data used to train the model 
 - folder "Pre_trained" holds the model training parameters.
 
 
 	2.Getting Started
         ----------------
 These instructions will get you a copy of the project up and running 
 on your local machine for testing purposes.
 
 * Prerequistes
 What things you need to install the software and how to install them.
 
 - Installing
 To run correctly this code please download and install anaconda 
 navigator installer from the website  below:
         	https://anaconda.org/anaconda/python
 
 - Libraries
 
 To install libraries open your Anaconda Prompt from the start menu
 and use the following command to install libraries:
     	    pip install library_name
 Below the list of libraries to install:
 numpy
 pandas
 sklearn
 matplotlib
 tensorflow 
 keras

 - IDE
 To execute the code use Spyder which is already included in Anaconda

 - Compatibility
 python 3.6

 - Set working directory

 Set the working directory to the folder that holds data

 	3. Usage
	 -------
 First import libraries and make sure that libraries are loaded correctly

 You can run the code as user or for building model.

 * To Train the model: run Part 1: Building model  
       		           Part 2: Model evaluation to evaluate model performances 
	 		   Part 3: Model Summary to get informations of execution details
	 		   Part 6: Confusion matrix to evaluate classifier performances
 N.B: After training model you can use the Part 4 to save trained model parameters.

* To use the model as user for prediction: you should load a pretrained model that is already trained
						and saved in your hard disc.
		      run Part 5: Load Pre-trained model
    4. Results
	----------
Performance analysis has shown that the pre-trained model achieve an accuracy (the number of correct predictions from all 
predictions made) of 90% on training data, and 84% on test data, it's capable to classify 
correctly 167 observations from 200. 










