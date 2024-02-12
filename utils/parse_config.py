from utils.tools import dotdict
from datetime import datetime
import json

class ParseConfig():
    def __init__(self, config):
        self.config = config
        self.dtypes_dict = {}

    def str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")

    def format_converter(self, the_dict, key, val, section =None):
        # convert to the required format
        if section is None:
            if key in self.dtypes_dict["int_dtypes"]:
                the_dict[key] = int(val)
            elif key in self.dtypes_dict["date_dtypes"]:
                the_dict[key] = datetime.strptime(val, '%d/%m/%Y')
            elif key in self.dtypes_dict["float_dtypes"]:
                the_dict[key] = float(val)
            elif key in self.dtypes_dict["bool_dtypes"]:
                the_dict[key] = self.str2bool(val)
            elif key in self.dtypes_dict["list_int_dtypes"]:
                the_dict[key] = [int(i) for i in val.split(',') if i]
            elif key in self.dtypes_dict["list_bool_dtypes"]:
                the_dict[key] = [bool(i) for i in val.split(',') if i]
            elif key in self.dtypes_dict["list_str_dtypes"]:
                the_dict[key] = [i for i in val.split(',') if i]
            else:
                the_dict[key] = val
        else:
            if key in self.dtypes_dict["int_dtypes"]:
                the_dict[section][key] = int(val)
            elif key in self.dtypes_dict["date_dtypes"]:
                the_dict[section][key] = datetime.strptime(val, '%d/%m/%Y')
            elif key in self.dtypes_dict["float_dtypes"]:
                the_dict[section][key] = float(val)
            elif key in self.dtypes_dict["bool_dtypes"]:
                the_dict[section][key] = self.str2bool(val)
            elif key in self.dtypes_dict["list_int_dtypes"]:
                the_dict[section][key] = [int(i) for i in val.split(',') if i]
            elif key in self.dtypes_dict["list_bool_dtypes"]:
                the_dict[section][key] = [bool(i) for i in val.split(',') if i]
            elif key in self.dtypes_dict["list_str_dtypes"]:
                the_dict[section][key] = [i for i in val.split(',') if i]
            else:
                the_dict[section][key] = val
            return the_dict

    def as_dict(self):
        """
        Converts a ConfigParser object into a dictionary.

        The resulting dictionary has sections as keys which point to a dict of the
        sections options as key => value pairs.
        """
        # get the default section to know datatypes
        self.dtypes_dict = {}
        for key, val in self.config.items("dtypes"):
            # convert to the required format
            self.dtypes_dict[key] = [x for x in val.split(",") if x]

        # now parse config to correct format
        the_dict = dotdict()
        for section in self.config.sections():
            if section == "dtypes": # skip dtypes as we already extracted the details
                continue
            elif "model" in section: # if model pararmeters add section
                the_dict[section] = {}
                for key, val in self.config.items(section):
                    self.format_converter(the_dict, key, val, section)
            else: # for all non model parameters no section
                for key, val in self.config.items(section):
                    self.format_converter(the_dict, key, val)
        return the_dict

def get_dataset_feat(dataset):
    with open('configs/datasets.json') as json_file:
        data_args_dict = json.load(json_file)
    if dataset not in data_args_dict.keys():
        print('dataset '+dataset+' features missing.')
        data_args_dict[dataset] = None
    return data_args_dict[dataset]

def get_model_hps(model):
    """
    by default the models have 6 params 
    all parameters should have ~ seperating the grid values
    coma is used to indicate to the ini file that the argument is a list

    static params are defined as value:datatype
    """
    with open('configs/model_params.json') as json_file:
        model_params = json.load(json_file)
    return model_params[model]["dynamic_params"], model_params[model]["static_params"]