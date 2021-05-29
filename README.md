<h1 align=center>GHOST</h1>
<p align=center>
  <img src="https://image.freepik.com/free-vector/vector-illustration-cute-cartoon-halloween-ghost_43633-3344.jpg" width=150>
  <br>
<a href="https://github.com/anonymousalpaca/ghost-dl/blob/master/README.md">about</a>  :: 
<a href="https://github.com/anonymousalpaca/ghost-dl/">code</a>  ::
<a href="https://github.com/anonymousalpaca/ghost-dl/blob/master/LICENSE">license</a>  ::
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

## RQ1: Does standard deep learning work for defect prediction?

For defect prediction, standard deep learners usually do not perform better than existing state-of-the-art methods in 30/40 experiments.
[Code](./RQ1.ipynb)

## RQ2: Why does deep learning fail for defect prediction?

The lack-of-success of deep learning in defect prediction can be attributed to optimizing for the wrong performance measure.
[Code](./RQ2.py)

## RQ3: How might we improve the results of deep learners on defect prediction?

For most evaluation goals, this modified version of deep learning performs better than the prior state-of-the-art.
[Code](./RQ3.sh')

## RQ4: Does deep learning work well in all cases?

Depending on the goals of the data mining, deep learning may or may not be best choice.
[Code](./RQ4.sh)

## RQ5: How slow is tuning for deep learning for defect prediction?

Tuning deep learners is both practical and tractable for defect prediction.
[Link to code](./RQ5.py)
