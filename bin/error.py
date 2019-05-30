#  SCCS Keywords  "%Z% %M%  %I%  %H%"

# -*- coding: utf-8 -*-
"""
Disclaimer (to be modified): This code, provided by BCG, is a working prototype only. It is supplied "as is" without any
 warranties and support. BCG assumes no responsibility or liability for the use of this code. BCG makes no representation
  that this code will be suitable for use without further testing or modification.
"""

import datetime
import logging
import os
import sys
import traceback
from errno import EACCES, EPERM, ENOENT
from logging.handlers import TimedRotatingFileHandlerf

import configparser

# cx_Oracle 5.2.1
# matplotlib 2.0.0
# numpy 1.11.3
# pandas 0.19.2
# scikit-learn 0.18.1
# scipy 0.18.1
CX_ORACLE_V = "5.2.1"
NP_V = ["1.11.3", "1.12.0"]
PD_V = "0.19.2"
MATPLOTLIB_V = "2.0.0"
SKLEARN_V = "0.18.1"
SCIPY_V = "0.18.1"

def setup_logging(model_log):
    """ Carries out the setup for
    the logging function
    """

    #need individual portions to create the right date format
    date_month = datetime.datetime.today().strftime('%m')
    date_day = datetime.datetime.today().strftime('%d')
    date_year = datetime.datetime.today().strftime('%y')
    date_cur = date_month + date_day + date_year

    model_file = model_log + 'tp2_' + date_cur + '.log'

    if not os.path.exists(model_file):
        try:
            with open(model_file, "w"):
                pass
        except (IOError, OSError) as e:
            print_error_message(e, model_file)
            raise

    #logging.basicConfig(level=logging.INFO)
    #logging.propagate = False  # default is to not print out to system

    logger = logging.getLogger()
    handler = TimedRotatingFileHandler(model_file, when="D", interval=1, backupCount=0)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    #logger.info("Logging setup complete")
    return logger

def print_error_message(e, file_name, logger=None, message=True):
    if message:
        traceMsg = traceback.format_exc()
    else:
        traceMsg = ""

    if logger is None:de
        if hasattr(e, 'errno'):
            #PermissionError
            if e.errno == EPERM or e.errno == EACCES:
                print "Error 1.2a: Cannot read  }\n{3}".format(e.errno, e.strerror, file_name, traceMsg)

            #FileNotFoundError
            elif e.errno == ENOENT:
                print "Error 1.2a: Cannot read from file / folder {1} due to FileNotFoundError({0}) in module {2}\n{3}".format(e.errno, e.strerror, file_name, traceMsg)
            elif e.__class__ == IOError:
                print "Error 1.2a: Cannot read from file / folder {1} due to I/O error: {0}\n{2}".format(str(e), file_name, traceMsg)
            elif e.__class__ == OSError:
                print "Error 1.2a: Cannot read from file / folder {1} due to OS error: {0}\n{2}".format(str(e), file_name, traceMsg)
            else: #general error
                if hasattr(e, "strerror"):
                    print "Error: {0} in module {1}\n{2}".format(e.strerror, file_name, traceMsg)
                elif hasattr(e, "message"):
                    print "Error: {0} in module {1}\n{2}".format(e.message, file_name, traceMsg)
                else:
                    print "Error in module {0}\n{1}".format(file_name, traceMsg)
        else:
            if hasattr(e, "strerror"):
                print file_name + ": " + e.strerror + "\n"
                print traceMsg
            elif hasattr(e, "message"):
                   print file_name + ": " + e.message + "\n"
                print traceMsg
            ''l;'':
                print traceMsg
    else:
        if hasattr(e, 'errno'):
            # PermissionError
            if e.errno == EPERM or e.errno == EACCES:
                logger.error("Error 1.2a: Cannot read from file / folder {1} due to PermissionError({0}) in module {2}\n{3}".format(e.errno, e.strerror, file_name, traceMsg))

            # FileNotFoundError
            elif e.errno == ENOENT:
                logger.error("Error 1.2a: Cannot read from file / folder {1} due to FileNotFoundError({0}) in module {2}\n{3}".format(e.errno, e.strerror, file_name, traceMsg))
            elif e.__class__ == IOError:
                logger.error("Error 1.2a: Cannot read from file / folder {1} due to I/O error: {0}\n{2}".format(str(e), file_name, traceMsg))
            elif e.__class__ == OSError:
                logger.error("Error 1.2a: Cannot read from file / folder {1} due to OS error: {0}\n{2}".format(str(e), file_name, traceMsg))
            else:  # general error
                if hasattr(e, "strerror"):     
                    logger.error("Error: {0} in module {1}\n{2}".format(e.strerror, file_name, traceMsg))
                elif hasattr(e, "message"):
                    logger.error("Error: {0} in module {1}\n{2}".format(e.message, file_name, traceMsg))
                else:
                    logger.error("Error in module {0}\n{1}".format(file_name, traceMsg))
        else:
            if hasattr(e, "strerror"):
                logger.error(file_name + ": " + e.strerror + "\n")
                logger.error(traceMsg)
            elif hasattr(e, "message"):
                logger.error(file_name + ": " + str(e.message) + "\n")
                logger.error(traceMsg)
            else:
                logger.error(traceMsg)


def createPaths(home, config):
    if home[-1] == '/' or home[-1] == '\\':
        home = home[:-1]

    paths_dict = {'c2p_path': home + config["PATHS"]["c2p_path"],
                  'p2c_path' : home + config["PATHS"]["p2c_path"],
                  'log_path' : home + config["PATHS"]["log_path"],
                  'error_log_path' : os.environ['LOGDIR'] + config["PATHS"]["error_log_path"],
                  'model_log_path' : home + config["PATHS"]["model_log_path"],
                  'model_path' : home + config["PATHS"]["MODEL_OBJ_PATH"],
                  'init_path' : home + config["PATHS"]["init_path"],
                  'input_path' : home + config["PATHS"]["INPUT_PATH"]}

    return paths_dict

def checkErrors(type="Consumer", test=False):
    ck_err = False
    error_msg = ""
    success_msg = ""
    config = ""
    logger = ""
    home = "" 

    if type == "Consumer": #producer only needs config, logger, and paths check
        #Check library versions
        try:
            import cx_Oracle as cx
        except ImportError, e:
            error_msg += "Error 1.3a: Python library not found: cx_Oracle\n"
            ck_err = True
        else:
            v = cx.__version__
            if v == CX_ORACLE_V:
                success_msg += "cx_Oracle loaded with version: " + v + "\n"
            else:
                error_msg += "Error 1.3b: Python library loaded with the wrong version: cx_Oracle (" + v + ")\n"
                ck_err = True

        try:
            import matplotlib
        except ImportError, e:
            error_msg += "Error: 1.3a: Python library not found: matplotlib\n"
            ck_err = True
        else:
            v = matplotlib.__version__
            if v == MATPLOTLIB_V:
                success_msg += "matplotlib loaded with version: " + v + "\n"
            else:
                error_msg += "Error 1.3b: Python library loaded with the wrong version: matplotlib (" + v + ")\n"
                ck_err = True

        try:
            import numpy as np
        except ImportError, e:
            error_msg += "Error: 1.3a: Python library not found: numpy\n"
            ck_err = True
        else:
            v = np.__version__
            if v in NP_V:
                success_msg += "numpy loaded with version: " + v + "\n"
            else:
                error_msg += "Error 1.3b: Python library loaded with the wrong version: numpy (" + v + ")\n"
                ck_err = True

        try:
            import pandas as pd
        except ImportError, e:
            error_msg += "Error: 1.3a: Python library not found: pandas\n"
            ck_err = True
        else:
            v = pd.__version__
            if v == PD_V:
                success_msg += "pandas loaded with version: " + v + "\n"
            else:
                error_msg += "Error 1.3b: Python library loaded with the wrong version: pandas (" + v + ")\n"
                ck_err = True

        try:
            import sklearn
        except ImportError, e:
            error_msg += "Error: 1.3a: Python library not found: sklearn\n"
            ck_err = True
        else:
            v = sklearn.__version__
            if v == SKLEARN_V:
                success_msg += "sklearn loaded with version: " + v + "\n"
            else:
                error_msg += "Error 1.3b: Python library loaded with the wrong version: sklearn (" + v + ")\n"
                ck_err = True

        try:
            import scipy
        except ImportError, e:
            error_msg += "Error: 1.3a: Python library not found: scipy\n"
            ck_err = True
        else:
            v = scipy.__version__
            if v == SCIPY_V:
                success_msg += "scipy loaded with version: " + v + "\n"
            else:
                error_msg += "Error 1.3b: Python library loaded with the wrong version: scipy (" + v + ")\n"
                ck_err = True

    # environmental variables
    try:
        os.environ["TP2_HOME"]
    except KeyError:
        error_msg += "Error 1.4: Environmental variable $TP2_HOME not found.\n"
        ck_err = True
    else:
        home = os.environ["TP2_HOME"]
        success_msg += "Environmental variable $TP2_HOME found: " + home + "\n"

    dbEnvironment = ["DBUSER", "DBPWD", "ORACLE_SID", "LOGDIR"]
    for i in dbEnvironment:
        try:
            os.environ[i]
        except KeyError:
            error_msg += "Error 1.4: Environmental variable $" + i + " not found.\n"
            ck_err = True
        else:
            success_msg += "Environmental variable $" + i + " found.\n"

    if ck_err:
        return ck_err, success_msg, error_msg, logger, config, home  # failed due to no libraries or environment variable
    else:
        # Setup config
        try:
            config = configparser.ConfigParser()
            configCheck = config.read(home + "/bin/config.ini")

            if len(configCheck) != 1:
                error_msg += "Error 1.2a: Cannot read from file/folder: " + home + "/bin/config.ini \n"
                ck_err = True

                default = configparser.ConfigParser()
                default_check = default.read(home + "/bin/default.ini")

                if len(default_check) != 1:
                    raise
        except (IOError, OSError) as e:
            error_msg += "Error 1.2a: Cannot read from file/folder: " + home + "/bin/default.ini due to " + e.strerror + "\n"
            ck_err = True
        except Exception as e:
            error_msg += "Error 1.2a: Cannot read from file/folder: " + home + "/bin/default.ini due to " + str(e) + "\n"
            ck_err = True

    if ck_err:
        return ck_err, success_msg, error_msg, logger, config, home # failed due to no config
    else:
        # Load paths from config
        paths_dict = createPaths(home, config)

        if home[-1] == '/' or home[-1] == '\\':
            home = home[:-1]

        paths_dict['ipc'] = home + "/data/ipc"

        for path in paths_dict:
            try:
                if not os.path.exists(paths_dict[path]):
                    os.mkdir(paths_dict[path])
            except (IOError, OSError) as e:
                error_msg += "Error 1.2b: Cannot write to file/folder:  " + path + " due to :" + e.strerror + "\n"
                ck_err = True
            else:
                success_msg += "Directory " + paths_dict[path] + " is accessible\n"

        # Setup logging
        try:
            logger = setup_logging(paths_dict['error_log_path'])
        except (IOError, OSError) as e:
            error_msg += "Error 1.2b: Cannot write to file/folder: " + paths_dict['error_log_path'] + " due to " + e.strerror + "\n"
            ck_err = True
        else:
            success_msg += "Logging file found and logger started successfully \n"

    if ck_err:
        return ck_err, success_msg, error_msg, logger, config, home  # failed due to pathing
    else:
        # Check to ensure we have connectivity with request transmission system
        if not os.path.exists(paths_dict['p2c_path'] + "init"):
            try:
                with open(paths_dict['p2c_path'] + "init", "w"):
                    pass
            except (IOError, OSError) as e:
                error_msg += "Error 1.2a: Cannot read from file/folder: " + paths_dict['p2c_path'] + "init due to :" + e.strerror + "\n"
                ck_err = True
            else:
                success_msg += "File " + paths_dict['p2c_path'] + "init is accessible\n"
        else:
            success_msg += "File " + paths_dict['p2c_path'] + "init is accessible\n"

        if not os.path.exists(paths_dict['c2p_path'] + "init"):
            try:
                with open(paths_dict['c2p_path'] + "init", "w"):
                    pass
            except (IOError, OSError) as e:
                error_msg += "Error 1.2a: Cannot read from file/folder: " + paths_dict['c2p_path'] + "init due to :" + e.strerror + "\n"
                ck_err = True
            else:
                success_msg += "File " + paths_dict['c2p_path'] + "init is accessible\n"
        else:
            success_msg += "File " + paths_dict['c2p_path'] + "init is accessible\n"

    if type == "Consumer" and not test: #only for consumer and not in test mode
        try:
            dbuser = os.environ['DBUSER']
            dbpwd = os.environ['DBPWD']
            dbSID = os.environ['ORACLE_SID']
            host = config["DB"]["db_host"]
            host = dbSID + host

            db = cx.connect(dbuser, dbpwd, host)

        except Exception, e:
            error_msg += "Error 1.1: Oracle DB connection error:" + str(e) + "\n"
            ck_err = True
        else:
            success_msg += "Oracle connection successfully made.\n"
            db.close()

    return ck_err, success_msg, error_msg, logger, config, home


if __name__ == "__main__":
    # Check for catastrophic errors
    ck_err, success_msg, error_msg, logger, config, home = checkErrors()

    if ck_err:
        error_msg = "Startup encountered the following catastrophic errors:\n" + error_msg
        print error_msg
        success_msg = "Startup was able to perform the following:\n" + success_msg
        print success_msg
        sys.exit(1)
    else:
        success_msg = "Startup was able to perform the following:\n" + success_msg
        logger.info(success_msg)
        sys.exit(0)
