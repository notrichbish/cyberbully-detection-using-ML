# Expose Cyberbully Tweets Using Machine Learning: A Data Science Approach

## Abstract
The advancements of technology along with the digitization of the relationships made a great impact among the centennials to mandatorily maintain a social media account. Despite the entertainment that social media provides, cyberbullying has been identified as a real issue in Malaysia which makes many centennials as victims. However, a few studies have been reported in detecting the attempt of cyberbullying on social media. Therefore, a solution using suitable data science techniques which can detect the attempt of cyberbullying on social media would be ideal. This research used the suspicious tweets dataset from Kaggle to build three supervised learning predictive models namely Naïve Bayes, SVM, and LSTM and tuned using Random Grid Search and Keras tuner to indicate a suitable solution. As a summary, Naïve Bayes model performed the best in terms of both accuracy and area under the curve (AUC) values with 88.4% and 0.81 respectively. While the LSTM model achieved the second-best with an accuracy of 90.6% and an AUC value of 0.58. Hence, with a greater number of records, both the accuracy
and AUC values of the LSTM model can be improved. 

## Files 
The main EDA and model training are performed in the FYP.ipynb file

The main deployment file is the "FYP_Deployment.py" file

".streamlit" file is the customized theme for the deployment website.

"saved model" file contains the Naive Bayes model used for the Deployment and the Count Vectorizer save file.

"all_model" file contains all the model that has been trained and save into a pickle file.


## Steps to run the deployment
1. Download Anaconda navigator and set up environment with python 3.8.
2. Once set up, click on home, and launch VS Code. Ensure that the appropriate enviromnent selected in the applications on dropdown menu.
3. To download all the dependencies needed, run the following command in the terminal	
	pip install -r requirements.txt
4. To run deployment open new terminal and redirect the directory using "cd (deployment file path)"
5. Then, type on the terminal "streamlit run FYP_Deployment_NB.py"

This final year project has been officially published under Young Investors Journals (YIJ). Link: https://lnkd.in/gTNkCV8G
