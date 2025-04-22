#### Parking Violations

Using PySpark to answer this question: 
- When are tickets most likely to be issued?

#### Installation

To get this project working, you need to create a new directory to house the github in the already made "spark-examples" directory. Once the new directory is created, a git clone command is required. This github page houses all the necessary codes, and shell scripts for running all parts of the project.

Link to the GitHub: https://github.com/chriskuz/big_data_computing_lab2

To get the data required for the PySpark program, you need to traverse into big_data_computing_lab_2/src/parking. Once you are in this directory, run the grab_data.sh shell script. This shell script uses an API key to get the data directly from the source. It will then create a new directory called data, and store a file called parking_data.csv in that directory. This is the data that will be used in PySpark. 

Once you have the data and are ready to run the PySpark code, travel back into src/parking and locate the parking_script.sh file. This is the shell script for running the necessay components. If you receive a permission denied error, type in the following - chmod +x `<SHELL_FILE_NAME.sh>`. After running this command, you should be able to run the script with no issues.

(note: this program assumes you have the necessary libraries and installations on your cluster.)