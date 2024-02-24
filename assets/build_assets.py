#!/usr/bin/env python

import pathlib 
import os
import subprocess


root = pathlib.Path()

# https://stackoverflow.com/questions/653380/how-to-convert-a-pdf-to-png-with-imagemagick-convert-or-ghostscript
command = 'gs -sDEVICE=pngalpha -o images/zoefarmer_resume_%01d.png -r144 docs/zoefarmer_resume.pdf'

subprocess.run(command, shell=True)