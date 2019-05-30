#  SCCS Keywords  "%Z% %M%  %I%  %H%"

# -*- coding: utf-8 -*-
"""
Disclaimer (to be modified): This code, provided by BCG, is a working prototype only. It is supplied "as is" without any
 warranties and support. BCG assumes no responsibility or liability for the use of this code. BCG makes no representation
  that this code will be suitable for use without further testing or modification.
"""

from __future__ import division
import os
import sys
import cPickle as pickle
import uuid
import datetime
import time
import pandas as pd

os.path.dirname(os.path.abspath("__file__"))
PATHS = ['./data/inputs/']
for i in PATHS:
    sys.path.append(i)

from economic_model_support_functions import OptimalIncentives
from error import print_error_message, checkErrors, createPaths
from calibration_support_functions import init_file_parser, settings_from_init_file
from cwt_class import cwt_production_class

def main(config, logger, test=False):
    """
    This function loads all the models from memory and sets up the environment
    for receiving requests
    Note: all files in model_path need to be pickle files and all of them will be
    loaded into the model. If the model names are not in the INI file then
    they will not be used in prediction.
    """
    # Load paths from config
    home = os.environ[config["PATHS"]["HOME"]]

    # Load paths from config
    paths_dict = createPaths(home, config)

    ceilinglookup_filename = paths_dict['input_path'] + config["DATA"]["IWA_CEILING_PROD"]
    svc_to_prod_filename = paths_dict['input_path'] + config["DATA"]["SVC_MATCHING"]
    strategic_overlay_filename = paths_dict['input_path'] + config["DATA"]["STRATEGIC_OVERLAY"]
    sic_to_industry_filename = paths_dict['input_path'] + config["DATA"]["SIC_TO_INDUSTRY"]
    cwt_filename = paths_dict['model_path'] + config["MODELS"]["CWT"] + ".p"
    accessorial_filename = paths_dict['input_path'] + config["DATA"]["ACCESSORIAL"]
    accessorial_map = paths_dict['input_path'] + config["DATA"]["ACCESSORIAL_MAP"]
    datatypes_filename = paths_dict['input_path'] + config["DATA"]["DATA_TYPE"]

    # Load models from model_path directory
    model_objs = {}
    logger.info("Model Path loaded from: " + paths_dict['model_path'])

    for model in os.listdir(paths_dict['model_path']):
        try:
            if (model == "README.MD") or (model[-2:] != ".p") or (model == "cwt_production.p"):
                continue

            with open(paths_dict['model_path'] + model, "rb") as model_pickle:
                infoLine = "Loading " + model + " model..."
                logger.info(infoLine)

                modelName = model[:-2]
                model_objs[modelName] = pickle.load(model_pickle)
        except Exception, e:
            print_error_message(e, "Error 3.2a: Model cannot be loaded: " + model, logger)

    logger.info("All models loaded")

    # Load config variables
    settings = settings_from_init_file(init_file_parser(paths_dict['init_path']))

    # Start run() which will check for requests
    run(home, paths_dict['c2p_path'], paths_dict['p2c_path'], model_objs, settings, paths_dict['init_path'],
        ceilinglookup_filename, svc_to_prod_filename, strategic_overlay_filename, sic_to_industry_filename,
        datatypes_filename, cwt_filename, accessorial_filename, accessorial_map, test, logger)


def run(home, c2p_path, p2c_path, model_objs, settings, init_path, ceilinglookup_filename, svc_to_prod_filename,
        strategic_overlay_filename, sic_to_industry_filename, datatypes_filename, cwt_filename, accessorial_filename,
        accessorial_map, test, logger):

    """
    This function runs as continuous loop and receives and processes requests
    using the models brought into memory using the setup() function
    """
    modified_start = max([os.path.getctime(p2c_path + f) \
                          for f in os.listdir(p2c_path)])
    logger.info("Consumer up and running")

    try:
        #read in datatypes and create a dtypes dict
        datatypes_table = pd.read_csv(datatypes_filename, index_col='Feature')
        data_type_dict = datatypes_table.T.to_dict(orient='record')[0]

        ceilinglookup_table = pd.read_csv(ceilinglookup_filename,
                                          dtype={'Product': 'str', 'Min_List_Rev_Wkly': 'float64',
                                                 'Max_List_Rev_Wkly': 'float64', 'Off_Inc_Cap': 'float64'})
        svc_to_prod_table = pd.read_csv(svc_to_prod_filename, dtype=str)
        strategic_overlay_table = pd.read_csv(strategic_overlay_filename)
        sic_to_industry_table = pd.read_csv(sic_to_industry_filename, dtype=str)
        accessorial_table = pd.read_csv(accessorial_filename)
        accessorial_map = pd.read_csv(accessorial_map)
        #cwt calibration tables

        pd.options.mode.chained_assignment = None
        model = OptimalIncentives(settings=settings, model_objects=model_objs,
                                  ceilinglookup_file=ceilinglookup_table,
                                  svc_to_prod_file=svc_to_prod_table,
                                  industry_name_lookup=sic_to_industry_table,
                                  strategicOverlay=strategic_overlay_table,
                                  accessorial=accessorial_table,
                                  accessorial_map=accessorial_map,
                                  isProduction=True)

        model_cwt = cwt_production_class(cwt_filename, svc_to_prod_table, settings)
    except Exception, e:
        print_error_message(e, "Error 3.2b: Model created error", logger)
        raise

    while True:
        #print "checking: " + p2c_path
        request = check_requests(p2c_path, modified_start)
        if request:
            logger.info("Found request %s", str(request))
            for ff in request:
                if ff == "init":
                    continue

                try:
                    data, tp20_bid_shpr = extract_data_ipc_file(p2c_path + ff)
                    data = data.groupby('Product').first().reset_index()
                    data.to_csv('data.csv')
                    #tp20_bid_shpr.to_csv('shipping.csv')

                    for colname in data_type_dict.keys():
                        if colname in data.columns:
                            data[colname] = data[colname].astype(data_type_dict[colname])
                except EOFError as e:
                    logger.error(e, "Error 3.3a: Consumer.py crashed due to file access: " + e, logger)
                    pass
                except Exception as e:
                    logger.error(e, "Error 3.3b: Consumer.py crashed due to model run: ", logger)
                    pass
                else:
                    try:
                        # Run the prediction using loaded model
                        start_time = datetime.datetime.now()

                        non_cwt_data = data[~data.Product_Mode.isin(['AIR_CWT', 'GND_CWT'])]
                        cwt_data = data[data.Product_Mode.isin(['AIR_CWT', 'GND_CWT'])]

                        results = model.run_calculator_production(non_cwt_data, tp20_bid_shpr)

                        mode_list = data.Product_Mode.unique()

                        if 'AIR_CWT' in mode_list or 'GND_CWT' in mode_list:
                            results_cwt = model_cwt.scorer(cwt_data)
                            results = results.append(results_cwt)

                        time_elapsed = datetime.datetime.now() - start_time

                        logger.info("Time elapsed: " + str(time_elapsed))
                        # with open("time_elapsed", "a") as time_file:
                        #     time_file.write(str(time_elapsed) + " - " + str(request) + "\n")

                        send_response(results, c2p_path)
                    except (IOError, OSError) as e:
                        logger.error(e, "Error 3.3a: Consumer.py crashed due to file access: " + e, logger)
                        pass
                    except ValueError as e:
                        logger.error(e, "Error 3.3b: Consumer.py crashed due to model run: ", logger)
                        pass
                    except Exception as e:
                        logger.error(e, "Error 3.3b: Consumer.py crashed due to model run: ", logger)
                        pass

            modified_start = max([os.path.getctime(p2c_path + f) \
                          for f in os.listdir(p2c_path)])
        time.sleep(1) # Sleep for 1 second

def extract_data_ipc_file(file_path):
    """Extracts data from a pickles file at file_path
    Returns the data"""
    num_tires = 10
    while True:
        try:
            with open(file_path, "rb") as pickle_file:
                data, tp20_bid_shpr = pickle.load(pickle_file)
            break
        except EOFError:
            if num_tires == 0:
                raise
            else:
                #"EOFError on file: " + str(file_path) + " trying again..."
                num_tires = num_tires - 1
                continue
        except (IOError, OSError) as e:
            print_error_message(e, file_path)
            raise
        except Exception as e:
            raise
    return data, tp20_bid_shpr

def send_response(data, folder_path):
    """
    Write requests to unique file in folder_path
    """
    file_name = str(uuid.uuid4())
    with open(folder_path + file_name, 'wb') as pickle_file:
        pickle.dump(data, pickle_file, pickle.HIGHEST_PROTOCOL)

def check_requests(folder_path, modified_start):
    """
    This functions checks to see if there are any pending requests to process
    and returns a list of files to process
    """
    all_files = {f:os.path.getctime(folder_path + f) for f in os.listdir(folder_path)}
    unprocessed = [f for f in all_files.keys() if all_files[f] > modified_start]
    if unprocessed:
        return unprocessed
    else:
        return False

if __name__ == "__main__":
    # Check for catastrophic errors
    if len(sys.argv) > 1 and sys.argv[1] == '-t':  # anything in the argument assumes test run
        cError, successMsg, errorMsg, logger, config, home = checkErrors("Consumer", True)
    else:
        cError, successMsg, errorMsg, logger, config, home = checkErrors()

    if cError:
        print "Startup encountered the following errors:"
        print errorMsg
        print "Startup was able to perform the following:"
        print successMsg
        sys.exit(1)
    else:
        logger.info("Consumer started")
        logger.info(successMsg)

        if len(sys.argv) > 1 and sys.argv[1] == '-t':  # anything in the argument assumes test run
            main(config, logger, test=True)
        else:
            main(config, logger)
