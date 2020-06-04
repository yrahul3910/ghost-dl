# GHOST
GHOST (Goal-oriented Hyperparameter Optimization for Scalable Training) is a paradigm for fast training and tuning of deep learners for software engineering. We use DODGE for hyperparameter optimziation; as such, the code is based off the [DODGE repository](https://github.com/amritbhanu/Dodge). 

# Usage

- Use `pip install -r requirements.txt` to install all the python3 dependencies and package (alternatively, run `make install`).
- Run `python3 main_[d2h|popt20].py`

# Research Questions

## RQ1: Can deep learners be used in software engineering without unreasonable training times?

The median training times for all datasets on a CPU were less than 2 seconds.
[Link to code](./doc/RQ1.py)

## RQ2: Can we perform hyperparameter optimization for deep learners in software engineering?

We were able to find optimal hyperparameters for our deep learners in under 4 minutes on average using DODGE.
[Link to code](./doc/RQ2.py)

## RQ3: Can we tune deep learners to optimize for specific metrics?

Our experiments clearly show an ability to tune the network to optimize for specific metrics by tuning the loss function.
[Link to code](./doc/RQ3.py)

## RQ4: Are deep learners scalable?

Our models scale well with the size of our datasets. With a 10x increase in the size of the data, the largest increase in the runtime was less than 4x.
[Link to code](./doc/RQ4.py)
