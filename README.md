# Kaggle_Competition：Spaceship-Titanic
- **Rank：209/2,289(9.13%)**
- **Target：Predict whether travelers will be transported to other dimensions(binary classification).**
- **Models：SVC、naive_bayes、LogisticRegression、KNN、DecisionTree、RandomForest、GradientBoost、AdaBoost、XGBoost**
## Introduction：
### Finding of EDA:  
1. **Proportion of Target Variables:**
   - The distribution of the target variable (Transported) is nearly balanced between "Yes" and "No."

2. **Age Distribution:**
   - The highest proportion of individuals transported falls in the age group under 18 years old. Conversely, the smallest proportion is observed in the age group between 18 and 40 years old.

3. **Service Preferences:**
   - A significant number of individuals seem to refrain from purchasing additional services and are likely from Earth.

4. **Marital Status Impact:**
   - The majority of single travelers appear to be more likely candidates for transportation.

### Data Engineering:
1. **PassengerID Transformation:**
   - **New Feature:** Group unit indicating family members, including the sequential number of family members and a flag for whether they are single.

2. **Cabin Transformation:**
   - **New Features:** Deck, Side, and Number extracted from the original cabin feature.

3. **Age Transformation:**
   - Ages are categorized into six groups based on age ranges.

4. **Consumption-related Features Transformation:**
   - **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck:**
     - Divided into four intervals based on consumption amount, determined by Median and Mean calculations.

**Total Features Added: 29**

## Conclusion
Upon comparing the performance across nine different models, it is evident that XGBoost and RandomForest outperform the others. Further optimization using GridSearchCV for parameter tuning results in the best overall performance. The lowest validation loss is achieved by implementing a Stacking approach, combining XGBoost and RandomForest models.

	   
	   
