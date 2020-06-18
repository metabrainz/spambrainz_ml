# Dummy data generator 

The current files are used to generate dummy data to be fed into LodBrok model for testing/training puposes. The file spam_dummy_data_generator and non_spam_dummy_data_generator notebook are used to create the eleven paramaters of spam and non-spam editor accounts respectively required for LodBrok, namely: 
* email
* website
* bio
* area
* privs
* gender
* birth_date
* member_since
* email_confirm_date
* last_updated
* last_login_date 

The file uses the spam and non-spam domains of emails and websites obtained thorugh research and colloboration. Rest all parameters are purely synthetic which are genrated through analysis of editor accounts. All the parameters are generated and stored in a pickle file to be used later to train the model.

The test data generated is also kept in this folder, which has 10000 values each for spam and non_spam datasets. In that, 8000 values are used to train the model and 500 values are used to test the trained model. The data generated are stored in pickle files which take about 100 MB of data storage.

### How to use:

Create a virtual environment and install all the libraries listed in requirements_dummy_data.txt.
```
pip3 install -r requirements_dummy_data.txt
```
Run the jupyter notebook in virtual environment:
```
jupyter notebook
```