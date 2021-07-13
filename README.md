# polarimeter_dev
Software for laser polarimeter data acquisition system.
This project is written in Python, so you need to install Python 3.7 and all required packages.
There is a simple way to do this using the Miniconda package manager.
##Getting Miniconda
Download the latest version of miniconda from  <br /> https://docs.conda.io/en/latest/miniconda.html </br>
<br /> bash Miniconda3-py39_version.sh < /br>
During the installation process, you are required to initialize conda by editing the .bashrc file. You need to type <br />conda init<br /> to make it automatically 
Make sure that your conda version is the latest: type <br /> conda update conda <br />
##Installing polarimeter software
Make a local copy of the repo: <br />git clone https://github.com/zakharov-binp/polarimeter_dev</ br>
Inside the polarimeter_dev type: <br />conda env create -f pol_env.yml<br />. This creates conda enviroment with all required packeges
To stark working with polarimeter software activate the enviroment: <br />conda activate pol_env <br />
##Running the code

Processing data from coordinate detector is double-step operation. At the first step raw data files from the detector need to be preprocessed. 
Before running the script, you need to edit configuration file: pol_config.yml 
To run the preprocessing type <br />python pol_preprocess.py <br />
This produces detector hitmaps and saves them as a set of histogramms in .npz file
By running <br /> python pol_fit.py --config pol_fit_config.yml --regex_line '*'<br /> you can start the fitting script.
For further instructions look at the code)


