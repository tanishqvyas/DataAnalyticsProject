import shutil
import os

class Preprocessor:

    def __init__(self):
        pass    	


    # Function to save modified dataset as a new version of DataFrame
    def save_file(self, df, data_path):

        if os.path.exists(data_path):
            shutil.rmtree(data_path)

        os.mkdir(data_path)
        df.to_csv(data_path, index = False)


    # function to replace the Strings with respective numbers
    def change_col_val_string_to_numeric():
        pass  




if __name__ == '__main__':
    pass

