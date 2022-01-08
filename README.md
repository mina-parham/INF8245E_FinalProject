# MAP3.0

This project was coded in a modular format with utility scripts for ease of mainenance, understandability and reusability. For the ability to execute them in one shot, all the utility scripts were copied to `final.ipynb`.

### Steps to reproduce submission file on Kaggle

* Assuming the dataset is present in Kaggle environment`/kaggle/input` *

- Upload notebook in `final.ipynb` to Kaggle
- Execute all cells sequentially
- Prediction file with timestamp can be found in `/kaggle/output` directory

### Steps to reproduce submission file locally

- Update `PROJECT_HOME` variable to point root of project
- Download and store the train, test data in the directory `$PROJECT_HOME/input/fall2021-inf8245e-machine-learning` as `x_train.pkl, y_train.pkl, x_test.pkl`
- Notebook is present in `$PROJECT_HOME/code/final.ipynb`
- Execute all cells sequentially
- Prediction file with timestamp can be found in `$PROJECT_HOME/output` directory
