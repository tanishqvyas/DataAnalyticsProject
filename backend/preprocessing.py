import shutil
import os

class Preprocessor:

    def __init__(self):
        pass

    # This function fetches all the values in a column and returns the list
    def get_column_as_list(self, dataFrame, col_name):
        
        col_list = list(dataFrame[col_name].tolist())

        return col_list	


    # Function to save modified dataset as a new version of DataFrame
    def save_file(self, df, data_path):

        df.to_csv(data_path, index = False)


    # function to replace the Strings with respective numbers
    def change_col_val_string_to_numeric(self, dataFrame, col_name, mapping_dict):
        
        # Fetching the current list
        col_list = self.get_column_as_list(dataFrame, col_name)

        changed_list = []

        for i in col_list:

            changed_list.append(mapping_dict[i])

        dataFrame[col_name] = changed_list

        return dataFrame



if __name__ == '__main__':
    pass

