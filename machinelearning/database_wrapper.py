# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
from datetime import datetime
import os
import traceback


class Database:
    """
    Subsystem 2 database interface - uses the shared transformerDB.db.

    This wrapper
