import os
import sys

# Ensure the parent directory is in the system path so we can import app.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
