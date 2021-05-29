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

# On the Value of Oversampling for Deep Learning in Software Defect Prediction

This repository is the reference implementation for the paper in the title above.

GHOST (Goal-oriented Hyperparameter Optimization for Scalable Training) is a paradigm for fast training and tuning of deep learners for software engineering. We use DODGE for hyperparameter optimziation; as such, the code is based off the [DODGE repository](https://github.com/amritbhanu/Dodge). We use our lab's internal package, raise-utils, for standardized, high-quality code.

# Usage

- Use `pip install -r requirements.txt` to install all the python3 dependencies.
- Run `python3 steps.py`

Within `steps.py`, you will see a dictionary of options that can be turned on or off. This was used to perform the ablation study. You can choose how you wish to run GHOST using this dictionary as well as the helper functions below. To run the Scott-Knott tests, we used our package's implementation (see the docs [here](https://raise.readthedocs.io/en/latest/index.html#module-raise_utils.interpret)). 

# Paper

Our paper was accepted to IEEE Transactions on Software Engineering 2021. You can view it [here](https://ieeexplore.ieee.org/iel7/32/4359463/09429914.pdf?casa_token=BvjobEj94EYAAAAA:JfWkU3SXbqkM4suLSPUlCHF7zX3o9-T-ezVuivzDT8Dn1y0Nu7eT3bXh0uexTI9s9DEgQ_5u).

# Cite this

```
@article{yedida2021value,
  title={On the Value of Oversampling for Deep Learning in Software Defect Prediction},
  author={Yedida, Rahul and Menzies, Tim},
  journal={IEEE Transactions on Software Engineering},
  year={2021},
  publisher={IEEE}
}
```


