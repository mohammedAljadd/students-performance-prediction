# Students-performance-and-difficulties-prediction

   This project is about using machine learning algorithms to predict whether or not a student would pass the final exam. 
   The data set used is from UCI Machine Learning Repository.

<div style="text-align:center;padding-top:20px" >
    <img src="https://blog.kinems.com/content/images/2018/04/Tracking_Headline.png" />
</div>



National institute of posts and telecommunications <a href="http://www.inpt.ac.ma/" target="_blank">INPT</a> Rabat, Morocco.


<div style="text-align:center;padding-top:20px">
    <img src="http://www.inpt.ac.ma/sites/default/files/logo.png" />
</div>

<h4>Project contributors :</h4>

 * [AL JADD Mohammed](https://github.com/mohammedAljadd) <br/>
 * [BOUJIDA Hafssa ](https://github.com/hafssaboujida)    <br/>
 * [EL NABAOUI Nouhaila](https://github.com/Elnabaouinouhaila) 

<h4> Project advisor :</h4>

 * [Prof. Amina Radgui](https://www.linkedin.com/in/amina-radgui-88017424/) <br/>

 
# Introduction

  Education is an important element of the society, every government and country in the world work so hard to improve this sector. With the corona-virus outbreak that has disrupted life around the globe in 2020, the educational systems have been affected in many ways; studies show that student’s performance has decreased since then, which highlights the need to deal with this problem more seriously and try to find effective solutions, as well as the influencing factors.  

# Motivation

  The educational systems need, at this specific time, innovative ways to improve quality of education to achieve the best results and decrease the failure rate. 
  
  As students of the IT department who studied in the last month a little about machine learning, we know that in order for an institute to provide quality education to learners, deep analysis of previous records of the learners can play a vital role, and wanted to work on this challenging task. 

# Problematic

As already mentioned, with the help of the old students records, we can came up with a model that can let us help students improve their performance in exams by predicting the student success. So, it is obvious it's a problem of classification , and we will classify a student based on his given informations, and we will also use diffrent classifiers such as KNN or SVM classifier and compare between them. Many factors affect a student performance in exams like famiily problems or alcohol consumption, and by using our skills in machine learning we want to : </br></br>

	    1) predict whether a student will pass his final exam or not.
	    2) came up with the best classifier that is more accurate and avoid 
	    overfitting and underfitting by using simple techniques.
	    3) know what the most factors affect a student performance.

So, teachers and parents will be able to intervene before students reach the exam stage and solve the problems.


# Dataset processing

Now before training our model we have to process our data :

1) We have to map each string to a numerical value so that it can be used for model training.
       
       Let's take an example to make this clear : The mother job column contain five values : 'teacher', 
       'health','services', 'at_home' and 'other'.
      
       Our job then is to map each of these string to numerical values, and this is how it's done :
       df['Mjob'] = df['Mjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
        
    </br>    
        
        
2) We have to perform feature scaling :

        	Feature scaling is a method used to normalize the range of independent variables 
		or features of data. In data processing, it is also known as data normalization 
		and is generally performed during the data preprocessing step.
        
		This will allow our learning algorithms to converge very quickly. The operation requires 
		to take each column, let's say 'col', and replace it by :
        
    ![\Large](https://latex.codecogs.com/svg.latex?\Large&space;col=\frac{col-mean(col)}{max(col)})
    
    		But this is not the only scaling you can do, you can try also :
		
    ![\Large](https://latex.codecogs.com/svg.latex?\Large&space;col=\frac{col-mean(col)}{std(col)}), where **std**  refers to standard deviation.
    
:warning: **We will not perform feature scaling for binary columns that contains 0's and 1's** .


# Dataset visualization


# Model evaluation (metrics) :


Before starting training our classifers let us define the metrics that we will use to compare between our three classifiers :

    1) Confusion matrix :
    
<img src='https://miro.medium.com/max/2102/1*fxiTNIgOyvAombPJx5KGeA.png' width='440cm'>

    2) F1 score :
    
<img src = 'https://www.gstatic.com/education/formulas2/-1/en/f1_score.svg' width='440cm'>

TP = number of true positives <br>
FP = number of false positives <br>
FN = number of false negatives <br>

    3) The roc curve : A receiver operating characteristic curve, or ROC curve, 
    is a graphical plot that illustrates the diagnostic ability of a binary 
    classifier system as its discrimination threshold is varied. 


<img src ='https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Roc-draft-xkcd-style.svg/800px-Roc-draft-xkcd-style.svg.png' width='500cm'>

<br>
Now let's start by the first learning algorithm :
<br>

    4) ROC score : it's simply the value of the area under the roc curve. The best value is 1 because the area of 1x1 square is 1.

# Logistic regression


# KNN


# SVM

Now we will use Support Vector Machine algorithm and see how it will act on our data. But, let's define what is svm algorithm

    In machine learning, support-vector machines are supervised learning models with 
    associated learning algorithms that analyze data for classification 
    and regression analysis.

 It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs. We will use three kernel : Linear, polynomial and gaussian kernel.

    1) Linear kernel : Linear Kernel is used when the data is Linearly separable, that is, 
    it can be separated using a single Line. It is one of the most common kernels to be used. 
    It is mostly used when there are a Large number of features in a particular Data Set.
    
    2)  the polynomial kernel is a kernel function commonly used with support vector machines and 
    other kernelized models, that represents the similarity of vectors in a feature space over 
    polynomials of the original variables, allowing learning of non-linear models.

    3) Gaussian RBF(Radial Basis Function) is another popular Kernel method used in SVM models for more. 
    RBF kernel is a function whose value depends on the distance from the origin or from some point. 
    Gaussian Kernel is of the following format:
    
<img src='https://miro.medium.com/max/336/1*jTU-kuAWMnMMYwBWj8mTVw.png' width='360cm'>

    Using the distance in the original space we calculate the dot product (similarity) of X1 & X2.
    Note: similarity is the angular distance between two points.


Here are some plots for the three kernels that we've talked about :


    
<img src='https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_0011.png' width='460cm'>
<br><br>


Now after we train our three models, here are the metrics and the ROC plots for each svm kernel :

But let's talk about some parameters that are using in SVM :

1) <img src="https://latex.codecogs.com/svg.latex?\Large&space;C" /> : The C parameter tells the SVM optimization how much you want 
to avoid misclassifying each training example.

2) <img src="https://latex.codecogs.com/svg.latex?\Large&space;D" /> : It's basically the degree of the polynomial used to find the hyperplane to split the data.

3) <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma" /> : Intuitively, the gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.

:warning: We will use diffrent values of those svm parameters using a for loop and choose the ones that minimize the cost <img src="https://latex.codecogs.com/svg.latex?\Large&space;J_{val}" /> on the cross validation set.

So here is how it will be done:

Repeat{<br> <br>
        - choose c <br>
        - Train the svm model <br>
        - calculate <img src="https://latex.codecogs.com/svg.latex?\Large&space;J_{val}" /> <br>
}

After that we will be able to get the optimal value of c.

**Our results :**

    1) Linear kernel : svm.svc(C)

After training our first svm model, here are our results :   

| Metric            | Value         |
| -------------     |:-------------:|
| training time     | 16ms          |
| accuracy          | 82.8125  %    |
| f1 score          | 0.75          |
| confusion matrix  |[9 10]<br>[1 44]|
|The roc_auc_score  |0.7257309941520469	|


ROC Curve :

<img src='https://github.com/mohammedAljadd/Students-performance-and-difficulties-prediction/blob/main/plots/rocL.PNG' width='450cm'>


    2) Polynomial kernel : svm.svc(C,d)
    
Here we will use diffrent values for both C and d. For example if we have 2 values for C and 3 values for d then we have 2x3 possibilities.

After training our second svm model, here are our results :   

| Metric            | Value         |
| -------------     |:-------------:|
| training time     | 0ms          |
| accuracy          | 73.4375 %    |
| f1 score          |  0.6529505582137161  |
| confusion matrix  |[9 11]<br>[6 39]|
|The roc_auc_score  |0.643859649122807	|


ROC Curve :

<img src='https://github.com/mohammedAljadd/Students-performance-and-difficulties-prediction/blob/main/plots/rocP.PNG' width='450cm'>



    3) Gaussian kernel : svm.svc(C,gamma)
    
Here we will use diffrent values for both C and gamma.

After training our last svm model, here are our results :   

| Metric            | Value         |
| -------------     |:-------------:|
| training time     | 16ms          |
| accuracy          | 78.125 %    |
| f1 score          |  0.6799999999999999  |
| confusion matrix  |[7 12]<br>[2 43]|
|The roc_auc_score  |0.6619883040935672	|


ROC Curve :

<img src='https://github.com/mohammedAljadd/Students-performance-and-difficulties-prediction/blob/main/plots/rocG.PNG' width='450cm'>


Now, after training our three svm models, it is better to group all the results so it can be easy to compare between them :


<br>

Now after we train our three models, here are the metrics and the ROC plots :


This table contains all metrics :


|<span style="color:red">Metric | <span style="color:red">Linear kernel |<span style="color:red">polynomial kernel  |<span style="color:red">gaussian kernel|
| ----------------|---------------|-------------------|:-------------:|
| training time   |   82.8125     |     73.4375       |      78.125   |
| accuracy %      | 82.8125       |   73.4375         |    78.125     |
| confusion matrix|[9 10]<br>[1 44]| [8 11]<br>[6 39] |[7 12]<br>[2 43]|
| f1 score        |  0.75         |        0.65       |    0.68       |
| roc_auc_score   |  0.73         |      0.64         |      0.66     |





<img src ='https://github.com/mohammedAljadd/Students-performance-and-difficulties-prediction/blob/main/plots/compare_kernels.PNG' width='700cm'>

As you can see the training times are soo small thanks to feature scaling. As you can also notice, the best svm model the one that used the linear kernel. So let's show the results of this linear kernel svm model again :

**The training time** : <b><span style="color:red">16ms </span><br></b>
**The accuracy** : <b><span style="color:red">83.0 % </span><br></b>
**The f1 score** : <b><span style="color:red">0.75 </span><br></b>
**The roc_auc_score** : <b><span style="color:red">0.73</span><br></b>


As you can see the accuracy is pretty good for our problem, and the f1 score has a good value. The value of the area under the roc curve is approximatly 0.73 which is good.
The roc curve :

<img src ='https://github.com/mohammedAljadd/Students-performance-and-difficulties-prediction/blob/main/plots/rocLF.PNG' width='450cm'>



 
# Comparision


# Conclusion
