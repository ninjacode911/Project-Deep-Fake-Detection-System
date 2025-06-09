#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFake Detection System - Resource Compiler Output
Created on: 2025-06-05 14:30:00 UTC
Author: ninjacode911

This file is auto-generated from resources.qrc. DO NOT EDIT MANUALLY.
Contains compiled binary resources (icons, images etc).
"""

from PyQt6 import QtCore  # Changed from PySide6 to PyQt6

qt_resource_data = b"\
\x00\x00\x00\xfe\
<\
?xml version=\x221.\
0\x22 encoding=\x22UTF\
-8\x22?>\x0a<svg width\
=\x2224\x22 height=\x2224\
\x22 viewBox=\x220 0 2\
4 24\x22 fill=\x22none\
\x22 xmlns=\x22http://\
www.w3.org/2000/\
svg\x22>\x0a    <path \
d=\x22M6 9L12 15L18\
 9\x22 stroke=\x22#8B9\
49E\x22 stroke-widt\
h=\x222\x22 stroke-lin\
ecap=\x22round\x22 str\
oke-linejoin=\x22ro\
und\x22/>\x0a</svg>\
"

qt_resource_name = b"\
\x00\x05\
\x00o\xa6S\
\x00i\
\x00c\x00o\x00n\x00s\
\x00\x10\
\x0e\x17\x06\x87\
\x00c\
\x00h\x00e\x00v\x00r\x00o\x00n\x00-\x00d\x00o\x00w\x00n\x00.\x00s\x00v\x00g\
"

qt_resource_struct = b"\
\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x01\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x02\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x03\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x10\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
\x00\x00\x01\x97H\xb0v8\
"

def qInitResources():
    try:
        QtCore.qRegisterResourceData(0x03, qt_resource_struct, qt_resource_name, qt_resource_data)
    except Exception as e:
        print(f"Failed to initialize resources: {e}")

def qCleanupResources():
    try:
        QtCore.qUnregisterResourceData(0x03, qt_resource_struct, qt_resource_name, qt_resource_data)
    except Exception as e:
        print(f"Failed to cleanup resources: {e}")

qInitResources()
