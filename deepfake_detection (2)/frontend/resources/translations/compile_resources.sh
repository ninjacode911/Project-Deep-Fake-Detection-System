#!/bin/bash

# Compile Qt resources
pyside6-rcc resources.qrc -o resources_rc.py

# Make executable 
chmod +x compile_resources.sh