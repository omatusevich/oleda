```python
import oleda
import pandas as pd
df=pd.read_csv('titanic.csv')
oleda.report(df,target='Survived')
```
<hr>



<h2 align="center">Missed values</h2>


    
      
              Missing  % of Total
    Cabin         687        77.1
    Age           177        19.9
    Embarked        2         0.2
    
     
    



<hr>



<h2 align="center">Shap values</h2>


    
      



![png](output_6_7.png)



![png](output_6_8.png)



![png](output_6_9.png)



![png](output_6_10.png)



![png](output_6_11.png)



![png](output_6_12.png)



![png](output_6_13.png)



![png](output_6_14.png)



![png](output_6_15.png)



![png](output_6_16.png)



![png](output_6_17.png)


    
     
    



<hr>



<h2 align="center">Features</h2>


    
      
    
     



<hr>



<h3 align="center">Sex</h3>


    
     
                                 
    Type :            Categorical
    Distinct count :            2
    Missed %:                   0
    
     



![png](output_6_25.png)


    Anova passed for  ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']



![png](output_6_27.png)


    
     



<hr>



<h3 align="center">Pclass</h3>


    
     
                             
    Type :            Numeric
    Distinct count :        3
    Missed %:               0
    
     



![png](output_6_32.png)


    
     



<hr>



<h3 align="center">Fare</h3>


    
     
                             
    Type :            Numeric
    Distinct count :      248
    Missed %:               0
    
     



![png](output_6_37.png)



![png](output_6_38.png)


    
     



<hr>



<h3 align="center">Age</h3>


    
     
                             
    Type :            Numeric
    Distinct count :       89
    Missed %:         19.8653
    
     



![png](output_6_43.png)



![png](output_6_44.png)


    
     



<hr>



<h3 align="center">PassengerId</h3>


    
     
                             
    Type :            Numeric
    Distinct count :      891
    Missed %:               0
    
     



![png](output_6_49.png)



![png](output_6_50.png)


    
     



<hr>



<h3 align="center">Embarked</h3>


    
     
                                 
    Type :            Categorical
    Distinct count :            4
    Missed %:            0.224467
    
     



![png](output_6_55.png)


    Anova passed for  ['Survived', 'Pclass', 'Parch', 'Fare']



![png](output_6_57.png)


    
     



<hr>



<h3 align="center">SibSp</h3>


    
     
                             
    Type :            Numeric
    Distinct count :        7
    Missed %:               0
    
     



![png](output_6_62.png)


    
     



<hr>



<h3 align="center">Parch</h3>


    
     
                             
    Type :            Numeric
    Distinct count :        7
    Missed %:               0
    
     



![png](output_6_67.png)


    
     



<hr>



<h3 align="center">Cabin</h3>


    
     
                                 
    Type :            Categorical
    Distinct count :          148
    Missed %:             77.1044
    
     



![png](output_6_72.png)


    Anova passed for  ['Pclass', 'SibSp', 'Fare']



![png](output_6_74.png)


    
     
    



<hr>



<h2 align="center">Pearson correlations</h2>


    
      



![png](output_6_79.png)


    
     
    



<hr>



<h2 align="center">Cramers V staticstics</h2>


    
      



![png](output_6_84.png)


    
     
    



<hr>



<h2 align="center">Top correlated features</h2>


    
      





['Sex', 'Pclass', 'Fare', 'Age', 'PassengerId', 'Embarked', 'SibSp', 'Parch', 'Cabin']




![png](output_6_90.png)

