#!/bin/bash

# Compile Qt resources
pyside6-rcc resources.qrc -o resources_rc.py

# Compile translations
pyside6-lrelease translations/en_US.ts -qm translations/en_US.qm

echo "Resources compiled successfully!" 