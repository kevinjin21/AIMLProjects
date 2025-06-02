import logging
from pathlib import Path

class RecordLogs:

    def __init__(self, version, log_path, file_name):
        self.version = version
        self.log_path = log_path
        self.file_name = file_name
        self.log()

    def log(self):
        logging.basicConfig()
        logger = logging.getLogger(self.file_name)
        logger.setLevel(logging.DEBUG)

        # create a file handler
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.DEBUG)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s \t: %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the file handler to the logger
        logger.addHandler(handler)
        logger.info('Starting a New Run .........................')
        logger.info(f'Version number : {self.version}')

        self.logger = logger
