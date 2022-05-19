import sys
import logging


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(filename='example.log')],
    encoding='utf-8', 
    level=logging.DEBUG
)

logging.info('hello')
