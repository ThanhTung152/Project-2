# Disaster Response Pipeline Project

## Introduction
This project is a part of the Udacity's Data Scientist Nanodegree Program
![Udacity](https://mma.prnewswire.com/media/1121585/Udacity_Logo.jpg)

In this project, we turned raw data into meaningful results. First from two raw data files, we processed them into a cleaned data table and then passed it through the ML pipeline to classify the input text into different features and create a model to classify the data. text in and create a web page to do just that

## File Descriptions
### Folder: app
**run.py**: python script to launch web application.<br/>
Folder: templates: web dependency files (go.html & master.html) required to run the web application.

### Folder: data
**disaster_messages.csv**: real messages sent during disaster events<br/>
**categories.csv**: categories of the messages<br/>
**disaster_messages.csv**: real messages sent during disaster events<br/>
**categories.csv**: categories of the messages<br/>
**process_data.py**: ETL pipeline used to load, clean, extract feature and store data in SQLite database<br/>
**ETL Pipeline Preparation.ipynb**: Jupyter Notebook used to prepare ETL pipeline<br/>
**InsertDatabseName.db**: cleaned data stored in SQlite database

### Folder: models
**train_classifier.py**: ML pipeline for load cleaned data, train model and save trained model as (.pkl) file for later use<br/>
**classifier.pkl**: file contains trained model<br/>
**ML Pipeline Preparation.ipynb**: Jupyter Notebook used to prepare ML pipeline

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Running on http://192.168.1.3:3001/

## Result
### Run ETL pipeline, ML pipeline and run file run.py
![Image1](https://github.com/ThanhTung152/Project-2/blob/main/image/1.jpg)
![Image2](https://github.com/ThanhTung152/Project-2/blob/main/image/2.jpg)
![Image3](https://github.com/ThanhTung152/Project-2/blob/main/image/3.jpg)

### Run Web
![Image4](https://github.com/ThanhTung152/Project-2/blob/main/image/4.jpg)
![Image5](https://github.com/ThanhTung152/Project-2/blob/main/image/5.jpg)
