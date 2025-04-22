
# NBA Shot Data README

By now you should have thoroughly folllowed the instructions found at the root hierarchy of this repository. If you have not done so, please return to the root README and ensure your dependencies were installed and that your repository is housed in the proper location of your Fordham cluster. 

This README is meant to guide on through the usage of the NBA MapReduce programs that answer specific questions around the use of [this Kaggle dataset found here](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs)

# Sub-project Focus Question

This directory will be used to answer the following question.

- for each player, we define the comfortable zone of shooting as a matrix of {SHOT_DIST, CLOSE_DEF_DIST, SHOT_CLOCK}. Develop a MapReduce-based algorithm to classify each player's records into 4 comfortable zones. Considering the hit rate, which zone is the best for James Harden, Chris Paul, Stephen Curry, and Lebron James. 

# Setup

## Data
The dataset for this project is "small" and when decompressed, will only take up approximatly 16MB of space. Therefore, it is most probable to download the dataset and run locally via RAM.

If data were to be manually moved from your local setup to your cluster setup, it's important that one refers to the `.gitignore` to understand where data MUST be housed. The `data` folder must be housed at the root hierarchy of the repository and to leverage PySpark here correctly, the `csv` file must be named `shot_logs.csv`.

If planning to download the data directly, it is possible by running the `get_data.sh` shell script which will autopopulate a `data` folder at the proper hierarchy and insert the decompressed data in such folder. 

**You must have a valid Kaggle API key (`JSON`) set up properly in the `root` directory of your cluster. Please follow the instructions provided by [Kaggle's documentation for API key setup](https://www.kaggle.com/docs/api). The API JSON file should be installed in your `root` user as a global attribute to be accessible. It should not be housed in the your `username` elevated from `root`. For our project, we downloaded the `JSON` locally and uploaded to server via `ssh` to the manager node's `username` `/tmp` directory. We then signed into `root` and were able to move the `JSON` in accordance to Kaggle's recommended setup so that the `get_data.sh` script worked.**

## Security Access

Once the API key is properly setup, provided elevated security access to the repository's shell scripts by running on each shell script:

```
chmod +x <SHELL_FILE_NAME.sh>
```

... where `<SHELL_FILE_NAME.sh>` would be substituted with the actual name of the shell script. 


# Running

Running the PySpark program is simplistic. All outputs would be submitted through terminal. 

All shell scripts should be run within this `nba` directory after the `data` folder is housing `shot_logs.csv` 

**Running Scripts per Question**

1) Question 1: `./comfort_zones.sh`

A full video demonstration for this lab will be available at this link:
https://drive.google.com/drive/folders/1T61-F5ybL2MrzHhIQ0zTORkDHzEQy2Ws?usp=sharing

