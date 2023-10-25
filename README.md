# Kaggle_Competition：Spaceship-Titanic
### Rank：212/2,198(9.64%)
### Target：Predict whether travelers will be transported to other dimensions(binary classification).
### Models：SVC、naive_bayes、LogisticRegression、KNN、DecisionTree、RandomForest、GradientBoost、AdaBoost、XGBoost
## Introduction：
－Finding of EDA:  
	1. Proportion of target variables(Transported) in Yes/No is nearly.  
	2. The biggest proportion of age in be transported is under 18 years old, on the other hand, the smallest  proportion of age in be transported is between 18 and 40 years old.  
 	3. Most of people won't buy other services and from earth.  
  	4. Majority of single traveler can be transported.  

－Data Engineering:  
	1. PassengerID -> New feature: group unit,which family members,the number of family members in sequence and whether they are single.  
	2. cabin -> New feature: Deck,Side and Number  
	3. Age -> Divided into 6 groups according to age range.  
	4. RoomService、FoodCourt、ShoppingMall、Spa、VRDeck -> Divided into 4 intervals according to consumption amount*by Median and Mean*.
A total of 29 features have been added.  

## Conclusion
After Comparing with the effect in 9 models can found XGboost,RandomForest better and use GridSearchCV to adjust parameters for best result. finally, I can get lowest validation_loss by trying Stacking XGBoost and RandomForest model. 

	   
	   
