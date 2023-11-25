import logging
import os
from datetime import datetime

from functions import constants

def create_log_file(name):
    logging.basicConfig(
        filename=os.path.join(os.getcwd(), constants.LOG_FOLDER + datetime.now().strftime("%d%m%Y_%I%M%p") + '_' + name+'.log'),
        filemode='w', format='%(asctime)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
