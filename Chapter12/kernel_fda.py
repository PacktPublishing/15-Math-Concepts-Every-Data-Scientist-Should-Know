import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import polynomial_kernel


# Class to allow us to do kernel Fisher Discriminant Analysis (KFDA) 
# using pure polynomial kernels of the form (x^Ty)^p
#
# We'll base our code on the tutorial by Ghojogh et al, which  
# can be found at https://arxiv.org/pdf/1906.09436.pdf
class KFDA_Poly:
    
    def __init__(self, degree=1):
        self.degree = degree # Degree of polynomial kernel
        self.alpha = None # The training datapoint weights of the discriminant axis
        self.X_train = None # The feature vectors of the training data
        self.class_labels = None # The labels used for the two classes

        # The mean value of the projections onto the discriminant axis for training datapoints 
        # in the first class
        self.projected_mean_class1 = None 

        # The mean value of the projections onto the discriminant axis for training datapoints 
        # in the second class        
        self.projected_mean_class2 = None

        # Boolean flag to indicate whether optimal alpha weights have 
        # been calculated from the training data
        self.is_trained = False


    def fit(self, X, y, mu=1.0e-5):
        '''
        Function to fit the alpha weights of the discriminant line

        :param X: A pandas dataframe containing the feature vectors of the datapoints
        :type X: pandas DataFrame

        :param y: A pandas series containing the class labels of the datapoints
        :type y: pandas Series

        :param mu: The strength of the diagonal contribution to the ???
                   covariance matrix. This regularizes and stabilizes the 
                   estimate of the covariance matrix.
        :type mu: float

        '''

        # Find unique class labels
        classes = y.unique()
        self.class_labels = classes
    
        # Identify rows of the training data that correspond to 
        # the first class
        idx_class1 = y==classes[0]

    	# Extract the total number of training datapoints
    	# and the number of training datapoints in each class
        n_data = X.shape[0]
        n1 = np.sum(idx_class1)
        n2 = X.shape[0] - n1
    
    	# Calculate the sub-Gram matrices for each class
        K1 = polynomial_kernel(X=X, Y=X.loc[idx_class1,:], degree=self.degree, gamma=1.0, coef0=0.0)
        K2 = polynomial_kernel(X=X, Y=X.loc[~idx_class1,:], degree=self.degree, gamma=1.0, coef0=0.0)
    
    	# Create scaled vectors of all ones, as we will
    	# use them repeatedly for aggregation
        v1 = np.ones(n1)/float(n1)
        v2 = np.ones(n2)/float(n2)

    	# Create matrices for mean centering
        H1 = np.identity(n1) - (float(n1)*np.outer(v1, v1))
        H2 = np.identity(n2) - (float(n2)*np.outer(v2, v2))
    
    	# Calculate vector of differences in class means
        deltaM = np.matmul(K2, v2) - np.matmul(K1, v1)

        ## Now calculate the common within class covariance matrix

        # First we'll add the regularization term
        N = np.identity(n_data)*mu
    
    	# Now we'll add the contributions from the training data 
    	# from each class
        N += np.matmul(np.matmul(K1, H1), np.transpose(K1))
        N += np.matmul(np.matmul(K2, H2), np.transpose(K2))

         # Now calculate the optimal weights
        self.alpha = np.matmul(np.linalg.inv(N), deltaM)

        # Calculate the mean projections of each class onto the trained Fisher axis
        self.projected_mean_class1 = np.mean(np.matmul(self.alpha, K1))
        self.projected_mean_class2 = np.mean(np.matmul(self.alpha, K2))
    
        # Set the other fields that we will need for 
        # making predictions
        self.X_train = X
        self.is_trained = True

    
    def predict(self, X):
        '''
        Function to make predictions using the trained kernel Fisher Discriminant

        :param X: A pandas dataframe containing the feature vectors of the datapoint 
                  for which predictions are required. One row is one datapoint.
        :type X: pandas DataFrame

        :returns: A list of the predicted class labels
        :rtype: python list
        '''
        
        # Check if the kernel Fisher Discriminant has been trained
        if self.is_trained:

            predicted_class_labels = [1]*X.shape[0]

            for i in range(X.shape[0]):
                # Project data point onto the Fisher axis
                projection = np.matmul(self.alpha,
                	                   polynomial_kernel(X=self.X_train, Y=X.iloc[i,:].to_numpy().reshape(1,-1),
                	                   degree=self.degree,
                	                   gamma=1.0,
                	                   coef0=0.0))
    
                # Calculate the distances along the Fisher axis to the mean of each class in the training data
                dis1 = np.abs(projection - self.projected_mean_class1)
                dis2 = np.abs(projection - self.projected_mean_class2)
        
                # See which class mean is closest to the projection of our datapoint
                # and make prediction of class label accordingly
                if dis1 < dis2:
                    predicted_class_labels[i] = self.class_labels[0]
                else:
                    predicted_class_labels[i] = self.class_labels[1]
                    
            return predicted_class_labels        
        else:
            print("Kernel Fisher Discriminant has not been trained")
            
            return None

                    
    def score(self, X, y_true):
        '''
        Function to score the predictive accuracy of the trained kernel
        Fisher Discriminant

        :param X: A pandas dataframe containing the feature vectors of the datapoint 
                  for which predictions are required. One row is one datapoint.
                  :type X: pandas DataFrame

        :param y_true: A pandas series containing the ground-truth class labels
                       for the datapoints in X. The row order of y_true must
                       match the same row order as X.
        :type y_true: pandas Series

        :returns: The proportion of datapoints for which the predicted class label
                  was correct
        :rtype: float
        '''
    
        # Check if the kernel Fisher Discriminant has been trained
        if self.is_trained:
            predicted_labels = self.predict(X)
            count_correct = np.sum([predicted_labels[i]==y_true[i] for i in range(len(predicted_labels))])
            
            # Return fraction of correctly predicted class label
            return float(count_correct)/float(X.shape[0])
        else:
            print("Kernel Fisher Discriminant has not been trained")
