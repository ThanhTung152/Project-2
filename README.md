# Disaster Response Pipeline Project

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
