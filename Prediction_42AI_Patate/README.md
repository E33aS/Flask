## Tool Use

This is a little web tool made with Flask that:
1. predicts labels for a given image.
2. uses a given model previously trained with the training script found in the `train_model` folder.
3. compares to the true labels of the image if those are known.


## Environment

You need to run this tool in a conda environment that has the following libraries:
numpy, 
flask, 
json5, 
opencv, 
tensorflow

The file `create_env_flask.sh` creates this environment called `flask`.


## Run

The tool shoud be launched from the `Prediction_Web` directory.
To run the tool, type the following in command line:
`FLASK_APP=check_image_label.py flask run`
or :
`python check_image_label.py`

The tool requires:
- an image.
- an optional .json file containing the true labels of the image.
- the path to the model.
- the number of directions (3 or 5) used when the model was trained.



