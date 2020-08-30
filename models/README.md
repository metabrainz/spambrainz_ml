# Model training

- how to use? 
    ``` python3 lodbrok.py```

## Internal working

- Load the ```../dataspambrainz_dataset.pickle``` file and train the model with it.
```
m = get_model()
train_model(model, training_data, [tensorboard])
```
- Save the model obtained to be used later:
 ```
 model.save("lodbrok1.h5")
 ```
This will store the model in an h5 type file, which later can be used to load the model for evaluation. The summary of the trained model is as follows:
![](summary.png)


# Model Retraining 

- how to use? 
    ```python3 retrain.py```

## Internal working

- The model has a slow static learning rate to learn new spam features while still being able to remember the old non_spam features.

- The simulation is shown here by retraining the model with a high number of epochs indicating similar cases.

- Rest is similar to how the original model is trained on and the model is saved as retrain_lodbrok.h5.

### Reasons to conclude this method:

- No need to store editor details for false-positive and false-negative cases respecting MBs data privacy rules.

- The model won't go through catastrophic forgetting (forget old learnings of what is spam or not) and will be able to learn new patterns in spam accounts over time.

- The structure of the data isn't changing over time (editor account fields remain the same).

- ##### Resources which helped me make this decision:
    * [Keras community help](https://github.com/keras-team/keras/issues/1868#issuecomment-191722497) (Discussion about the same exact problem [Online learning   in Keras for an LSTM model(LodBrok)])
    * [Stack overflow](https://stats.stackexchange.com/a/352771) (Explains about catastrophic forgetting and role of fit function)
    * [Machine Learning mastery blog](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/) (Explains the importance of learning rate on a model)
    * Reading research papers and consulting professors.




