#!/usr/bin/env python

"""config.py: """

__author__ = "Rajesh Thakur"
__copyright__ = "Copyright 2021, Zuma"
__created_date__ = "04-04-2022"

import os
from dotenv import load_dotenv, find_dotenv
print("yyyyy"+os.getcwd())
print(os.getcwd()+"/.environment")
load_dotenv(os.getcwd()+"/.environment")

BASE_DIR = os.getenv('BASE_DIR')

