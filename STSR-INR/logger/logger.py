import yaml
import os
import logging
import logging.config

def setup_logger(log_file_dir,log_config="./logger/logger_config.yml",replace=False):
    """set up the logger

    Args:
        log_file_dir (str): the dir path for log file to save (e.g. ".\Log")
        log_config (str, optional): the file path for the logger config file. Defaults to "./logger/logger_config.yml".

    Returns:
        logging: use logging.getLogger('LoggerName') to get the Logger you need
    ------------------------------------------------------------------------
        LoggerName:
            printLogger: logger used to print in console \n
            logger: logger used to save file in "log_file_dir/EasyLog.log"\n
            trainLogger: logger for train process,save file in "log_file_dir/info.log"\n
            utilsLogger: logger for utils function,save file in "log_file_dir/info.log"\n
            dataLogger: logger for dataio,save file in "log_file_dir/info.log"\n
            infLogger: logger for inference,save file in "log_file_dir/info.log"
    """
    logger_settings = None
    with open(log_config,'r') as f:
        logger_settings = yaml.safe_load(f)
        
        #get the file name in logger_config.yml
        file_console_filename = logger_settings['handlers']['file_console']['filename']
        easy_file_filename = logger_settings['handlers']['esay_file']['filename']
        
        #reset the log file path
        logger_settings['handlers']['file_console']['filename'] = os.path.join(log_file_dir,file_console_filename)
        logger_settings['handlers']['esay_file']['filename'] = os.path.join(log_file_dir,easy_file_filename)

        #if the info.log file exists, delete it
        if (os.path.exists(os.path.join(log_file_dir,file_console_filename)) and replace):
            os.remove(os.path.join(log_file_dir,file_console_filename))

    logging.config.dictConfig(logger_settings)
    
    return logging
