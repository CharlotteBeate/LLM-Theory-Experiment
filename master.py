################################################################################
# Master for the Agents in the Prior Work
################################################################################
# import numpy as np

import subprocess
import sys
# Setup and Confirmation and testing out
print("Setup and Confirmation and testing out")

scripts = ["transformed_softmax.py", "transformed_scalar.py"]
for script in scripts:
    subprocess.run([sys.executable, script])

## small examples
scripts = ["example2.py"]
for script in scripts:
    subprocess.run([sys.executable, script])



