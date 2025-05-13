# %%

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from dvc import api

# %%

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting plotting script")

# %%

# For each experiment variation load the results and plot them saving the outcome to plots folder

# %%
