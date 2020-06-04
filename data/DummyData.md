# Dummy data generator 

The current files are used to generate dummy data to be fed into LodBrok model for testing/training puposes. The file spam_dummy_data_generator notebook is used to create the eleven paramaters of spam editor accounts required for LodBrok, namely: 
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

The file uses the spam domains of emails and websites obtained thorugh research and colloboration. Rest all parameters are purely synthetic genrated through analysis of editor accounts. All the parameters are generated and stored in a pickle file to be used later.

### How to use:

Create a virtual environment and install all the libraries listed in requirements_dummy_data.txt.
```
pip3 install -r requirements_dummy_data.txt
```
Run the jupyter notebook in virtual environment:
```
jupyter notebook
```


