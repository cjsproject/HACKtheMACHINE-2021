# THIS IS AN EXAMPLE SCRIPT. YOU CAN USE THIS DIRECTLY OR ADAPT IT TO YOUR NEEDS.

'''
Hackthemacine 2021 Submission Script

This file is an example of the workflow expected by your
docker image submission.

Feel free to use this script directly or adapt it as needed for your solution.

What this script does:
1) load the test data (path to test data given as ENV variables)
2) perform inference on the test data
3) save your inference results to a csv (with the indicated CSV structure,
filename and destination path)

It doesn't matter what language or script your docker image uses as long as
it can accomplish those three steps.
'''
###################
# LIBRARY IMPORTS #
###################

# example inference script found at my_inference_model.py
from my_inference_model import MyClassifier

# functions used by the utility functions below
import pandas as pd
import os
import time

#################
# MAIN FUNCTION #
#################
'''
This main function will be what your docker container runs to generate
your submission CSV.

Again, just three steps your solution needs to perform:
1) Load the test data
2) Make predictions
3) Save your predictions as a CSV called "submission.csv"
'''
def main():

    # Step 1) load the test data
    test_data = load_test_data(data_type_requested="ember_flat")

    # Step 2) perform inference on the test data
    all_results = make_predictions(test_data)

    # Step 3) save your inference results to a .xlsx
    save_results(all_results)

#####################
# UTILITY FUNCTIONS #
#####################

'''
Utility function to perform STEP 1: "load the test data"

The test data your solution needs to run inference on as well as the output
submission file path can be found via system environment variables:

'EMBER_RAW_DATA_PATH' - test data in 'raw' EMBER features
'EMBER_FLAT_DATA_PATH' - test data in 'flattened' EMBER features
'OUTPUT_SUBMISSION_PATH' - where your "submission.csv" file should go
 '''
def load_test_data(data_type_requested):

    if data_type_requested=='ember_raw':
        ember_test_data_path = os.environ['EMBER_RAW_DATA_PATH'] # this ENV var is where our data is
        data = pd.read_excel(ember_test_data_path)
        data['data_type'] = 'ember_raw'

    elif data_type_requested=='ember_flat':
        ember_test_data_path = os.environ['EMBER_FLAT_DATA_PATH'] # this ENV var is where our data is
        data = pd.read_excel(ember_test_data_path)
        data['data_type'] = 'ember_flat'

    return data

'''
Utility function to perform STEP 2: "perform inference on the test data"

Here we call on the 'my_inference_model.py' script to perform our inference code.
'''
def make_predictions(test_data):

    model = MyClassifier() # refer to 'my_inference_model.py' for our example model object
    pre_data = model.preproc(test_data)

    all_results = []
    for i, sample in test_data.iterrows():
        result = {}
        result['sha256'] = sample['sha256']
        result['data_type'] = sample['data_type']
        result['pred'] = model.predict(pre_data[i].reshape(1, -1))
        all_results.append(result)

    return all_results

'''
Utility function to perform STEP 3: "save your inference results to a xlsx"

Two important things here:
1) Format of the XLSX should be correct.. contains 'sha256' and 'label' columns
2) Save to the correct spot (look for 'OUTPUT_SUBMISSION_DIR' env var) with correct name "submission.xlsx"
'''
def save_results(all_results):

    output_dir = os.environ['OUTPUT_SUBMISSION_PATH']

    all_results = pd.DataFrame(all_results)
    all_results.to_excel(f"{output_dir}/submission.xlsx",index=False)


# we're just going to run the main() function when this script is called by the Dockerfile
if __name__=='__main__':
    main()
