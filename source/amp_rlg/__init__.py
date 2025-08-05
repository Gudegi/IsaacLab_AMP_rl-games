import sys
import os

# Add the source directory to Python path to resolve amp_rlg imports
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.dirname(current_dir)
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)