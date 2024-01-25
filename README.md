# Breast-Cancer-Prediction
 Breast cancer prediction can be approached using machine learning algorithms. By analyzing a dataset that includes various features related to breast cancer, such as age, tumor size, and lymph node status, a predictive model can be built. 


The suggested method uses a trained deep learning neural network system to categorize breast cancer subtypes. According to data from 221 actual patients, the findings have an accuracy of 90.50 percent. Without needing any human intervention, this model can classify and identify breast cancer lesions.



![Preprocessing data](https://github.com/Rajendradegala/Breast-Cancer-Prediction/assets/140039152/a9fee7a7-a692-4433-be2a-1b14c398dcb2)

1.DATA  PREPROCESSING:

Data preprocessing is an essential step in working with datasets. It involves cleaning and transforming the data to make it suitable for analysis. This can include handling missing values, removing duplicates, scaling numerical features, encoding categorical variables, and more. By preprocessing the data, we can ensure that it is in a format that can be effectively used by machine learning algorithms. It's like preparing the ingredients before cooking a delicious meal! 🍳🥘


The Wisconsin Diagnostic Breast Cancer dataset is utilized, consisting of various features related to tumor characteristics. The dataset is explored, visualized, and divided into training and testing sets. A logistic regression model is trained and evaluated using accuracy metrics.


Step-by-step features processing:

Bra_Size: Although it looks numerical, it only ranges from 28 to 48, with most of the sizing lying around 34-38. It makes sense to convert this to categorical dtype. We'll fill the NA values into an 'Unknown' category. We can see above that most of the buyers have a bra-sizing of 34 or 36.


Bust- We can see by looking at the values which are not null, that bust should be an integer dtype. We also need to handle a special case where bust is given as - '37-39'.
We'll replace the entry of '37-39' with the mean, i.e.- 38, for analysis purposes. Now we can safely convert the dtype to int. However, considering that roughly 86% of the bust data is missing, eventually it was decided to remove this feature.


Category- none missing; change to dtype category.


Cup Size- Change the dtype to category for this column. This col has around 7% missing values. Taking a look at the rows where this value is missing might hint us towards how to handle these missing values.

mc_df.bra_size = mc_df.bra_size.fillna('Unknown')
mc_df.bra_size = mc_df.bra_size.astype('category').cat.as_ordered()
mc_df.at[37313,'bust'] = '38'
mc_df.bust = mc_df.bust.fillna(0).astype(int)
mc_df.category = mc_df.category.astype('category')
