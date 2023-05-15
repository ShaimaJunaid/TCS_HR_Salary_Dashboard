from flask import Flask,render_template,request
import pickle
import numpy as np

#instance of flask
app=Flask(__name__)
#load the model
model = pickle.load(open('model.pkl','rb'))

#Displaying the home page
@app.route('/')
def home():
   return render_template('home.html')

#displaying the predict Page
@app.route("/predict", methods = ["POST"])

#Define function predict
def predict():
    #converting the values to int and stored in features
    features = [int(x) for x in request.form.values()]
    print(features)
    #Convert features in to array
    features = [np.array(features)]
    print(features)
    #predict the values with features
    prediction = model.predict(features)
    print(prediction)
    #store the prection in output
    output = prediction
    #print(output)
    #if prediction is 0 then salary is less than 50K
    if(prediction[0] == 0):
      output = "Salary is less than 50K"  
    #if prediction is 1 then salary is more than 50K
    else:
       output = "Salary is more than 50K"
    #print(output)
    # Return the output to result page
    return render_template('result.html', prediction_text='{}'.format(output))
#for run the application  
if __name__=="__main__":
    app.run(debug = True)
    