import logging

# Create and configure logger
logging.basicConfig(filename='2048.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO
                    )
logger = logging.getLogger()  # Creating an object
logger.setLevel(logging.DEBUG)  # Setting the threshold of logger
