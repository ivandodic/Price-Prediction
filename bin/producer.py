#  SCCS Keywords  "%Z% %M%  %I%  %H%"

# -*- coding: utf-8 -*-
"""
Disclaimer (to be modified): This code, provided by BCG, is a working prototype only. It is supplied "as is" without any
 warranties and support. BCG assumes no responsibility or liability for the use of this code. BCG makes no representation
  that this code will be suitable for use without further testing or modification.

"""
import os
import time
import datetime
import uuid
import sys
import cPickle as pickle
from datetime import datetime
import pandas as pd

home = os.environ["TP2_HOME"]
os.path.dirname(os.path.abspath("__file__"))
paths = [home + '/data/inputs/', home + '/bin/']
for i in paths: sys.path.append(i)

from etl import pullData, pushData, transformData, create_prod_base_tables
from error import print_error_message, checkErrors, createPaths

def main(bid_number, config, logger, test=False, test_bids=5, timeout=30):
    """
    Main function, starts the program, setsup the paths and submits
    the bidnumber to process
    """
    pd.options.mode.chained_assignment = None

    # Load paths from config
    home = os.environ[config["PATHS"]["HOME"]]

    # Load paths from config
    paths_dict = createPaths(home, config)

    # Write initial error flag
    set_error_flag(bid_number, paths_dict['error_log_path'], str(1), logger)

    #to do bulk scoring: check tp_bid sample and get all unique bids
    bid_numbers = pd.DataFrame()
    if test:
        try:
            tp20_bid = pd.read_csv(home + '/data/tp_bid.csv', dtype=str)
            bid_numbers = tp20_bid['NVP_BID_NR'].unique()

            print "Bid numbers found: "
            print bid_numbers
        except RuntimeError as e:
            print_error_message(e, "Error 3.0: General producer error due to test run", logger, False)
            sys.exit(1)

    master = pd.DataFrame()
    master_result = pd.DataFrame()
    sql = ''

    if test:
        bids_to_score = test_bids
        if bids_to_score == -1:
            bids_to_score = len(bid_numbers)
    else:
        bids_to_score = 1

    for i in range(0, bids_to_score):
        if test:
            bid_number = str(bid_numbers[i])

        if test:
            logger.info("Processing bid # " + str(i+1) + " of " + str(len(bid_numbers)))

        logger.info("Processing bid: " + bid_number)

        # Get data
        try:
            response, tp20_bid_shpr, tp20_ceiling_svc, tncvcel, tp_accessorial = \
                get_data(home, bid_number, config, test, logger)
        except Exception, e:
            if test:
                continue
            else:
                #print_error_message(e, "Error 2.2a: Data transformation issues: ", logger)
                sys.exit(1)

        try:
            # Enqueue the data
            master = master.append(response)
            result = enqueue(response, tp20_bid_shpr, timeout, bid_number, paths_dict['c2p_path'],
                             paths_dict['p2c_path'], tncvcel, paths_dict['log_path'])
            master_result = master_result.append(result)
        except RuntimeError as e:
            print_error_message(e, "", logger, False)

            if test:
                continue
            else:
                sys.exit(1)

        try:
            # store data
            sql_result = put_data(home, bid_number, config, test, result, tp20_ceiling_svc, tp_accessorial, logger)
            if test:
                sql = sql + sql_result
            #True
        except Exception, e:
            print_error_message(e, "", logger, False)
            logger.warning("Bid " + bid_number + " scoring failed.")
        else:
            cleanup([paths_dict['p2c_path'], paths_dict['c2p_path']])
            logger.info("Bid " + bid_number + " successfully scored.")
            # Once done with everything write success flag
            set_error_flag(bid_number, paths_dict['error_log_path'], str(0), logger)
        finally:
            cleanup([paths_dict['p2c_path'], paths_dict['c2p_path']])

    if test:
        master.to_csv("master.csv")
        master_result.to_csv("results.csv")
        sql_file = open("sql_results.txt","w")
        sql_file.write(sql)
        sql_file.close()

def cleanup(file_paths):
    """Performs cleanup of working files"""
    # Delete IPC files
    for path in file_paths:
        files_to_delete = os.listdir(path)
        for file_name in files_to_delete:
            os.remove(path + file_name)

def set_error_flag(bid_number, log_path, flag, logger):
    try:
        with open(log_path + 'tp2_' + bid_number + '.log', 'w') as error_log:
            error_log.write(flag + "\n")
    except (IOError, OSError) as e:
        print_error_message(e, log_path, logger)
        raise


def get_data(home, bid_number, config, test, logger):
    """
    Gets the ETL'd data from controller
    """

    try:
        tp20_bid, tp20_bid_shpr, tp20_svc_grp, tp20_ceiling_svc,tp20_shpr_svc, ttpsvgp, zone_weight, tncvcel, \
        tp_accessorial = pullData(bid_number, config["DB"]["db_host"], home, test)
    except ValueError, e:
        print_error_message(e,
                            "Error 2.2b: Data transformation issue for bid number " + bid_number + " in step pullData",
                            logger)
        raise
    except Exception, e:
        print_error_message(e,
                            "Error 2.2a: Data transformation issue for bid number " + bid_number + " in step pullData",
                            logger)
        raise

    try:
        tp_bid_table, tp_bid_svc_table = transformData(tp20_bid, tp20_bid_shpr, tp20_svc_grp, tp20_shpr_svc, ttpsvgp)
    except Exception, e:
        print_error_message(e,"Error 2.2a: Data transformation issue for bid number "
                            + bid_number + " in step transformaData", logger)
        raise

    try:
        # add data check
        if zone_weight is None:
            prod_table = create_prod_base_tables(home, tp_bid_table, tp_bid_svc_table)
        else:
            prod_table = create_prod_base_tables(home, tp_bid_table, tp_bid_svc_table, zone_weight, tp20_svc_grp)
    except Exception, e:
        print_error_message(e, "Error 2.2a: Data transformation issue for bid number "
                            + bid_number + " in step createProdBaseTable", logger)
        raise

    prod_table = prod_table.fillna(0)

    return prod_table[prod_table.BidNumber == bid_number], tp20_bid_shpr, tp20_ceiling_svc, tncvcel, tp_accessorial

def put_data(home, bid_number, config, test, response, tp20_ceiling_svc, tp_accessorial, logger):
    """
    Places data into Oracle DB
    """

    try:
        response = response.merge(tp20_ceiling_svc, how='inner',
                                  left_on=['BidNumber', 'SVC_GRP_NR'], right_on=['NVP_BID_NR', 'SVC_GRP_NR'])

        if test:
            acy_table = response[response['Product_Mode'] == 'ACY']
            prod_table = response[response['Product_Mode'] != 'ACY']

            prod_table = prod_table[
                ["BidNumber", "Product", "Incentive_Freight", "Target_Low", "Target_High"]].drop_duplicates()

            for index, row in prod_table.iterrows():
                logger.info("{0}: Inc - {1}, Low - {2}, High - {3}".format(row["Product"], row["Incentive_Freight"],
                                                                           row["Target_Low"], row["Target_High"]))

            try:
                acy_table = acy_table[["BidNumber", "MKG_SVP_DSC_TE", "ASY_SVC_TYP_CD", "RESI", "DAS"]].drop_duplicates()

                for index, row in acy_table.iterrows():
                    if row["ASY_SVC_TYP_CD"] == 'RES':
                        logger.info("{0}: Inc - {1}".format(row["MKG_SVP_DSC_TE"], row["RESI"]))
                    elif row["ASY_SVC_TYP_CD"] == 'GDL':
                        logger.info("{0}: Inc - {1}".format(row["MKG_SVP_DSC_TE"], row["DAS"]))
            except Exception, e:
                pass

        return pushData(home, bid_number, config["DB"]["db_host"], config, response, tp_accessorial, logger, test)
    except (IOError, OSError) as e:
        print_error_message(e, home + "/" + bid_number+"_results.csv", logger)
        raise
    except RuntimeError, e:
        print_error_message(e,"",logger)
        raise


def send_results(output_data, tncvcel, bidNumber, log_path):
    """
    Combine the output and input and
    write to a pickle file
    """
    file_name = bidNumber + '-' + str(uuid.uuid4())

    acy = output_data[output_data['Product_Mode'] == 'ACY']
    reg = output_data[output_data['Product_Mode'] != 'ACY']

    # merge tncvcel to capture TP 1.0 values
    tncvcel = tncvcel.rename(columns={'NVP_BID_NR':'BidNumber', 'SVC_TYP_CD':'SVM_TYP_CD',
                                      'RCM_NCV_QY':'TP1_Target', 'NCV_MIN_QY':'TP1_Low', 'NCV_MAX_QY':'TP1_High'})

    output_data = reg.merge(tncvcel, how="inner")

    #re-add accessorials
    output_data = output_data.append(acy)

    #check is RESI and DAS are columns
    if 'DAS' not in output_data.columns:
        output_data['DAS'] = ''

    if 'RESI' not in output_data.columns:
        output_data['RESI'] = ''

    #add timestamp
    output_data['Timestamp'] = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')
    output_data.to_csv(log_path + file_name + '.csv')

    return output_data


def send_request(data, tp20_bid_shpr, folder_path, logger):
    """
    Write requests to unique file in folder_path
    """
    file_name = str(uuid.uuid4())

    #print "sending request: " + folder_path + file_name

    try:
        with open(folder_path + file_name, 'wb') as pickle_file:
            pickle.dump((data,tp20_bid_shpr), pickle_file, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print_error_message(e, "Error 2.2b: Data exchange issues writing " + folder_path + file_name, logger)
        raise

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

def extract_data_ipc_file(file_path, logger):
    """Extracts data from a pickles file at file_path
    Returns the data"""
    num_tires = 10
    while True:
        try:
            time.sleep(0.1)
            with open(file_path, "rb") as pickle_file:
                data = pickle.load(pickle_file)
            break
        except EOFError:
            if num_tires == 0:
                raise
            else:
                logger.info("EOFError on extracting: " + str(file_path) + " trying again...")
                num_tires = num_tires - 1
                continue
        except (IOError, OSError) as e:
            print_error_message(e, "Error 2.2b: Data exchange issues related to file: " + file_path, logger)
    return data

def enqueue(data, tp20_bid_shpr, timeout, bid_number, c2p_path, p2c_path, tncvcel, log_path):
    """
    Write the data to the Producer to Consumer file and check the Consumer to
    Producer file for the results
    """
    start_time = datetime.now()
    try:
        modified_start = max([os.path.getctime(c2p_path + f) \
                          for f in os.listdir(c2p_path)])
    except:
        modified_start = time.mktime(start_time.timetuple()) + start_time.microsecond / 1E6

    send_request(data, tp20_bid_shpr, p2c_path, logger)
    while True:
        if (datetime.now() - start_time).seconds > timeout:
            raise RuntimeError("Error 3.1: Producer.py timed out on bid number: " + bid_number)
        result_files = check_requests(c2p_path, modified_start)
        if result_files:
            for result_file in result_files:
                return send_results(extract_data_ipc_file(c2p_path + result_file, log_path),
                                    tncvcel, bid_number, log_path)
            break

if __name__ == "__main__":
    # Check for catastrophic errors
    cError, successMsg, errorMsg, logger, config, home = checkErrors("Producer")

    if cError:
        print "Startup encountered the following errors:"
        print errorMsg
        print "Startup was able to perform the following:"
        print successMsg
        sys.exit(1)
    else:
        args = len(sys.argv)

        # first argument is just the name of the file
        if args > 1 and args < 4:
            if args == 3:
                if sys.argv[1] == '-t':  # anything else in the INI assumes production run
                    #logger.propagate = True
                    input_command = "Test scoring:"

                    try:
                        default_bids = int(sys.argv[2])
                    except:
                        default_bids = 5

                    logger.info(input_command + " added to producer to run")
                    main(input_command, config, logger, True, default_bids)
                else:
                    logger.error( "Error 3.5: Unrecognized argument")
            else:
                try:
                    input_command = sys.argv[1]
                    try:
                        logger.info(input_command + " added to producer to run")
                        logger.info("Producer started")
                        main(input_command, config, logger)
                    except Exception, e:
                        print_error_message(e, "Error 3.0: General Bid Scoring failure", logger)
                except Exception, e:
                    print_error_message(e, "Error 3.4: No bid number provided", logger)

        else:
            logger.error("Error 3.4: No bid number provided")
            print "producer must have at least 2 arguments but no more than 3:"
            print "minimum: producer.py BIDNUMBER"
            print "test: producer.py -t NUMBER_OF_BIDS_TO_TEST"
            sys.exit(1)
