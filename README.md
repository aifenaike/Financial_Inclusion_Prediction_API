# Financial_Inclusion_Prediction_API
A Machine Learning System for predicting depression tendencies from socio-economic factors deployed as a RESTAPI using Python, regularized greedy forests, FastAPI, & NoSQL..


## Machine Learning Models as APIs
Application Programming Interfaces (or APIs for short) have been around for a long time and basically provide a standardized way of communication between two software applications that are not necessarily of the same type.

Applied to our context of data science, an API allows for the communication between a web page or app and our ML application. The API opens up certain user-defined URL endpoints, which can be used to send or receive requests with data. These endpoints are not dependent on the application: if you update your algorithm, the interface will stay the same. This minimizes the work required to update the running application.


## Novelty
Using APIs as a clean interface between the analytics and the application that makes use of them allows for faster product development and reusability of developed models in multiple applications. For instance, if a credit bank that wants to make use of the financial inclusion model. Instead of running this algorithm as part of a website/webapp, it runs as a separate service that the website front-end calls when needed. 
Now it’s clear why exposing your model’s code as an API makes it flexible and easy to integrate!


## Dataset
In this project, we will work with real-world data. The main dataset contains demographic information and what financial services are used by approximately 33,600 individuals across East Africa. This data was extracted from various Finscope surveys ranging from 2016 to 2018. 

## Results
The goal is to predict the likelihood of the person having a bank account or not (Yes = 1, No = 0), for each unique id belonging to individuals across four East African countries - Kenya, Rwanda, Tanzania, and Uganda. Evaluation measures for a binary classification model are used to assess how accurate prediictions are. Here I employed **The Mean Absolute Error (MAE)** as the evaluation metric. The MAE is a measure of how many scenarios were falsely claasified ( 1- Accuracy). The model was optimized towards reducing this measure of false prediction.
Using a seven fold stratified cross validation split, **I was able to obtain a Mean MAE score on test set: 0.11148367514540215. i.e an Accuracy of ~88.852%.**

## FastAPI
FastAPI is a fast web framework for building APIs with python, it comes with faster query time, easy and minimize code for you to design your first API within few minutes FastAPI. In this project we will use FastAPIand our prebuilt model to get inclusion tendencies from clients.

### Installing FastAPI

Installing FastAPI is the same as any other python module, but along with FastAPI you also need to install uvicorn to work as a server. You can install both of them using the following command:

```python
pip install fastapi uvicorn
```
