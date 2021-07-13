# polarimeter_dev
Software for laser polarimeter data acquisition system.
This project is written in Python, so you need to install Python 3.7 and all required packages.
There is a simple way to do this using the Miniconda package manager.
## Getting Miniconda
- Download the latest version of miniconda from   https://docs.conda.io/en/latest/miniconda.html 
- #+BEGIN_EXAMPLE bash Miniconda3-py39_version.sh #+END_EXAMPLE
- During the installation process, you are required to initialize conda by editing the .bashrc file. You need to type #conda init# to make it automatically 
Make sure that your conda version is the latest: type #conda update conda #
## Installing polarimeter software
Make a local copy of the repo: # git clone https://github.com/zakharov-binp/polarimeter_dev# </ br>
Inside the polarimeter_dev type: #+BEGIN_EXAMPLE conda env create -f pol_env.yml #+END_EXAMPLE This creates conda enviroment with all required packeges
To stark working with polarimeter software activate the enviroment: # conda activate pol_env # <br />
## Running the code
Processing data from coordinate detector is double-step operation. At the first step raw data files from the detector need to be preprocessed. 
Before running the script, you need to edit configuration file: _pol_config.yml_ 
To run the preprocessing type # python pol_preprocess.py #
This produces detector hitmaps and saves them as a set of histogramms in .npz file
By running # python pol_fit.py --config pol_fit_config.yml --regex_line # you can start the fitting script. </ br>
For further instructions look at the code)


