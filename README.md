# MetaVRF
This repository contains source code of the ICML 2020 paper:(Learning to Learn Kernels with Variational Random Features)
https://arxiv.org/abs/2006.06707


Learning to Learn Kernels with Variational Random Features
====================================================



The main components of the repository are:

* ``run_classifier.py``: script to run classification experiments on Omniglot and miniImageNet
* ``features.py``: deep neural networks for feature extraction and image generation
* ``inference.py``: amortized inference networks for various versions
* ``utilities.py``: assorted functions to support the repository
* ``train_regression.py``: script to run regression experiments.

Dependencies
------------
This code requires the following:

*  python 3
* TensorFlow v1.0+

Data
----
For Omniglot, miniImagenet, see the usage instructions in ``data/save_omniglot_data.py``, ``data/save_mini_imagenet_data.py``, respectively.

Usage
-----

* To run few-shot classification, see the usage instructions at the top of ``run_classifier.py``.
* To run view regression, see the usage instructions at the top of ``train_regression.py`` and  ``test_regression.py``.


Extending the Model
-------------------

There are a number of ways the repository can be extended:

* **Data**: to use alternative datasets, a class must be implemented to handle the new dataset. The necessary methods for the class are: ``__init__``, ``get_batch``, ``get_image_height``, ``get_image_width``, and ``get_image_channels``. For example signatures see ``omniglot.py``, ``mini_imagenet.py`` or ``omniglot.py``. Note that the code currently handles only image data. Finally, add the initialization of the class to the file ``data.py``.

* **Feature extractors**: to use alternative feature extractors, simply implement a desired feature extractor in ``features.py`` and change the function call in ``run_classifier.py``. For the required signature of a feature extractor see the function ``extract_features`` in ``features.py``.

Citation
---------
If MetaVRF is used in your paper/experiments, please cite the following paper.


@misc{zhen2020learning,

    title={Learning to Learn Kernels with Variational Random Features},
    
    author={Xiantong Zhen and Haoliang Sun and Yingjun Du and Jun Xu and Yilong Yin and Ling Shao and Cees Snoek},
    
    year={2020},
    
    eprint={2006.06707},
    
    archivePrefix={arXiv},
    
    primaryClass={cs.LG}
    
}

}
