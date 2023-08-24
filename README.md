# 專案名稱：Spaceship-Titanic
### 來源：Kaggle
### Kaggle排名：212/2,198(9.64%)
### 目標：預測旅客是否會被傳送到其他維度
### 比較模型：SVC、naive_bayes、LogisticRegression、KNN、DecisionTree、RandomForest、GradientBoost、AdaBoost、XGBoost
### 專案介紹：
EDA時發現 
1. PassengerID：可細分為第幾組、第幾個成員、家庭成員順序的數量及是否為單身。
2. cabin:區分為'Deck'、'Side'及Number 編號。
3. Age:依照年齡區間分6組。
4. RoomService、FoodCourt、ShoppingMall、Spa、VRDeck:以上欄位依照消費金額區分4個區間，*區分條件為中位數及平均數*。以上共新增至29個特徵。
  
比較上述9個模型成效可發現XGBoost、RandomForest表現最佳，使用GridSearchCV調參成效更佳，最後嘗試Stacking XGBoost及RandomForest model 可得最小的 validation_loss  

	   
	   
