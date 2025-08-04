# Lord of the Lattes: Exploring Alternate Universes in Fanfiction
This repository contains all the code for the Master's Thesis of Alina Kereszt, MSc Cognitive Science, Aarhus University. The thesis concerns the topic modeling of Alternate Universe fanfiction, and includes the code for scraping, preprocessing, topic modeling, and analyzing the data. Also included is the code for generating all the figures appearing in the thesis.

Contact me at 202005328@post.au.dk if you have any questions.

## Structure
The repository is structured as follows:

    ├── obj                                     <- Folder for the objects produced while running the project
    |   ├── data                                   <- Pandas dataframes in .pkl format
    |   ├── embeddings                             <- Embeddings in .pkl format
    |   ├── model_extras                           <- Saved objects related to machine learning models, e.g. test set, in .pkl format
    |   ├── models                                 <- Saved machine learning models in .pkl format
    |   ├── plots                                  <- All the plots produced during the analysis, in .png format
    |   ├── text_files                             <- Text outputs in .txt format
    |   ├── texts                                  <- Scraped fanfiction texts and metadata in .csv format
    |   └── topic_models                           <- Topic models in .pkl format
    ├── scrape                                  <- Folder for the script required to scrape fanfiction texts and metadata
    ├── src                                     <- Folder for the scripts required to run preprocessing, topic modeling and analysis
    |   └── utils                                  <- Helper functions and wrappers
    ├── .gitignore                              <- File to prevent backing up large files to cloud.
    ├── README.md                               <- README for the repository
    └── requirements.txt                        <- List of packages required for running the code

Note that not all folders in the repository are backed up to GitHub due to some large files being contained within. Please check the .gitignore file.

## Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.99.3, running Python version 3.13.3). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.

## Running the code
For scraping fanfiction metadata and text, Radiolarian's AO3 scraper was used (https://github.com/radiolarian/AO3Scraper). The commands for scraping can be found in the ```/scrape/scrape.sh``` file. Make sure to replace the header with your own ID for ethical scraping.

The scripts contained in ```/src/``` are numbered in the order they should be run in. Note that some may take a few hours to run.
