## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict Air Quality Index absed on training data in 'RealCombile.csv' file.
2. app.py - This contains Flask APIs that receives Air Quality details through API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted Air Quality Index.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py ( model can be anything like Linear Regression,Random Forest,Decision Tree,XgBoost etc)
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](Screenshots/Enter_Details.png)
Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
![alt text](Screenshots/Predicted.png)


