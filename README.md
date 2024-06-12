# Disaster Response Pipeline Project

## Project Overview
This project utilizes data engineering skills to analyze disaster data from Appen to build a model that classifies disaster messages into categories. This classification can help direct messages to the appropriate disaster relief agencies during crisis situations.

The repository includes a data cleaning pipeline, a machine learning pipeline to categorize messages sent during disasters, and a Flask web app where an emergency worker can input new messages and receive classification results across multiple categories. Additionally, the web app provides visual insights into the data.

## Repository Contents
- `data/`
  - `process_data.py`: Script to run the ETL pipeline that cleans data and stores it in an SQLite database.
  - `disaster_messages.csv`: Dataset including the disaster messages.
  - `disaster_categories.csv`: Dataset including the categories of the messages.
  - `DisasterResponse.db`: Created database from transformed and cleaned data.
- `models/`
  - `train_classifier.py`: Script to run the ML pipeline that trains the classifier and saves it.
  - `classifier.pkl`: Saved model file.
- `app/`
  - `run.py`: Flask file to run the web application.
  - `templates/`: Folder containing the HTML templates for the web app.
- `README.md`: Documentation of the project.

## Setup and Installation
Ensure you have Python 3.6+ installed along with the libraries mentioned in `requirements.txt`.

### Running the Scripts
1. **Run the ETL pipeline** to clean data and store it in a database:
   ```
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```
2. **Run the ML pipeline** to train the classifier and save it:
   ```
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

### Running the Web App
Navigate to the `app/` directory and run the following command to start the Flask web server:
```
python run.py
```

### Viewing the App
Go to [http://localhost:3000/](http://localhost:3000/) in your web browser.

## Instructions for Use
After setting up the database and model, and running the Flask app, you can:
- Enter a disaster message into the input box and click "Classify Message" to see the categories the message falls under.
- View visualizations of the dataset on the main page.

## Additional Notes
- The web app is intended to be simple. If you are comfortable with HTML, CSS, and JavaScript, feel free to enhance the web app to suit your needs.
- Effective use of comments and docstrings in the scripts helps explain the intention behind code blocks and the use of functions.

