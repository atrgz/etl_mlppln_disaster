# Disaster Response Pipeline

## Table of contents
- [Installations](#installations)
- [Project Motivations](#project-motivations)
- [Instructions](#instructions)
- [File Descriptions](#file-descriptions)
- [Licensing, Authors and Acknowledgments](#licensing-authors-and-acknowledgments)

## Installations
This project is written in Python 3.6.3 using the following  libraries:
* numpy 1.12.1
* pandas 0.23.3
* sqlalchemy 1.1.13
* nltk 3.2.5
* sklearn 0.19.1
* pickle 4.0
* re 2.2.1
* plotly 2.0.15
* flask 0.12.5

## Project Motivations
This work is the result of the project for the **Data Engineering** module of the [Data Scientist Nanodegree of Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025?promo=year_end&coupon=SAVE40&utm_source=gsem_brand&utm_source=gsem_brand&utm_medium=ads_r&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_campaign=19167921312_c_individuals&utm_term=143524475679&utm_term=143524475679&utm_keyword=udacity%20data%20science_e&utm_keyword=udacity%20data%20science_e&gad_source=1&gclid=EAIaIQobChMImKz0y_e0gwMVfj4GAB1FgAEHEAAYASAAEgI-h_D_BwE).

The goal of this project is to generate a ML model that predicts the categories of a text message. There is a list of 36 different categories, and each message can have multiple categories.

Specifically, the data used is from messages received during disasters. Categorizing this messages can help the disaster response teams as each category will be relevant for different teams. A good categorization of the messages can lead to quicker and more efective responses to disasters.

To achieve this goal, the data stored in CSV files is managed using an ETL pipeline that will result in a SQLite database. This database will be used to create, train and evaluate a ML model.

The results of the model and a few visualizations of the database can be seen in a web app created with Flask.

## Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Go to `app` directory: `cd app`

Run web app: `python run.py`

Click the `PREVIEW` button to open the homepage

## File Descriptions
The file structure of the projects is as follows:
'''
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # cleans data and stores it in a table called "Message" in a SQLite database
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # builds, trains and evaluates a model based in the database provided
|- classifier.pkl  # saved model
'''

Please note that if the database already have a table called "Message" an error will happen

## Licensing, Authors and Acknowledgments
This project has no specific license, but makes extensive use of the templates provided in the Udacity course.