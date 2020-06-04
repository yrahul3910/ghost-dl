<p align="center">
<b>GHOST</b> <br />
<a href="https://github.com/anonymousalpaca/ghost-dl/blob/master/README.md">about</a>  :: 
<a href="https://github.com/anonymousalpaca/ghost-dl/">code</a>  ::
<a href="https://github.com/anonymousalpaca/ghost-dl/blob/master/LICENSE">license</a>  ::
<a href="https://github.com/anonymousalpaca/ghost-dl/blob/master/INSTALL.md">install</a>  ::
<a href="https://github.com/anonymousalpaca/ghost-dl/blob/master/CODE_OF_CONDUCT.md">contribute</a>  ::
<a href="https://github.com/anonymousalpaca/ghost-dl/issues/">issues</a>  ::
<a href="https://github.com/anonymousalpaca/ghost-dl/blob/master/CONTACT.md">contact</a>
</p>
<p align="center">
<img src="https://img.shields.io/badge/language-python-orange.svg">&nbsp;
<img src="https://img.shields.io/badge/license-MIT-green.svg">&nbsp;
<img src="https://img.shields.io/badge/platform-mac,*nux-informational">&nbsp;
<img src="https://img.shields.io/badge/purpose-ai,se-blueviolet">&nbsp;
</p>
<hr />
# Why Deep Learning Fails for Defect Prediction (and How to Fix it using GHOST)
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

Our experiments clearly show an ability to tune the network to optimize for specific metrics by tuning the loss function. In `src/helper/ML.py`, in `run_model(...)`, change the fraction from `1.0 / frac` to `10.0 / frac` or `100.0 / frac` (the numerator is the weight; for unweighted, change the loss function to `binary_crossentropy`). Then, run the main code.
[Link to code](./src/defect prediction/main_d2h.py)

## RQ4: Are deep learners scalable?

Our models scale well with the size of our datasets. With a 10x increase in the size of the data, the largest increase in the runtime was less than 4x.
[Link to code](./doc/RQ4.py)
