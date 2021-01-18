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
 
# Introduction

  Education is an important element of the society, every government and country in the world work so hard to improve this sector. With the corona-virus outbreak that has disrupted life around the globe in 2020, the educational systems have been affected in many ways; studies show that student’s performance has decreased since then, which highlights the need to deal with this problem more seriously and try to find effective solutions, as well as the influencing factors.  

# Motivation

  The educational systems need, at this specific time, innovative ways to improve quality of education to achieve the best results and decrease the failure rate. 
  
  As students of the IT department who studied in the last month a little about machine learning, we know that in order for an institute to provide quality education to learners, deep analysis of previous records of the learners can play a vital role, and wanted to work on this challenging task. 

# Problematic

As already mentioned, with the help of the old students records, we can came up with a model that can let us help students improve their performance in exams by predicting the student success. So, it is obvious it's a problem of classification , and we will classify a student based on his given informations, and we will also use diffrent classifiers such as KNN or SVM classifier and compare between them. Many factors affect a student performance in exams like famiily problems or alcohol consumption, and by using our skills in machine learning we want to : </br></br>

	    1) predict whether a student will pass his final exam or not.
	    2) came up with the best classifier that is more accurate and avoid overfitting and underfitting by using 
	    simple techniques.
	    3) know what the most factors affect a student performance.

So, teachers and parents will be able to intervene before students reach the exam stage and solve the problems.


# Dataset processing

Now before training our model we have to process our data :

1) We have to map each string to a numerical value so that it can be used for model training.
       
       Let's take an example to make this clear : The mother job column contain five values : 'teacher', 'health',
       'services', 'at_home' and 'other'.
      
       Our job then is to map each of these string to numerical values, and this is how it's done :
       df['Mjob'] = df['Mjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
        
    </br>    
        
        
2) We have to perform feature scaling :

        	Feature scaling is a method used to normalize the range of independent variables or features of data. 
        	In data processing, it is also known as data normalization and is generally performed during 
		the data preprocessing step.
        
		This will allow our learning algorithms to converge very quickly. The operation requires to take each column, 
		let's say 'col', and replace it by :
        
    ![\Large](https://latex.codecogs.com/svg.latex?\Large&space;col=\frac{col-mean(col)}{max(col)})
    
    		But this is not the only scaling you can do, you can try also :
		
    ![\Large](https://latex.codecogs.com/svg.latex?\Large&space;col=\frac{col-mean(col)}{std(col)}), where **std**  refers to standard deviation.
    
:warning: **We will not perform feature scaling for binary columns that contains 0's and 1's** .


# Dataset visualization


# Logistic regression


# KNN


# SVM



# Comparision


# Conclusion
