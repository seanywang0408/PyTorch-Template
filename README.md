# Template

A template for develop a project in PyTorch. All the concern is about running code with lowest probability of making mistakes. 

## Concerns

To devise a clear coding structure, the code is divide into four folders, `configs`, `models`, `mylib`, and `scripts`. Besides, the cache and logs file would be store in the `tmp` folder, which is created at the beginning of running the code.

### `configs`

* Model configurations (batch size, epoch, model channels, and so on) and environments (data paths, checkpoint path and so on) are demonstrated in `configs`, so that one can easily change the configs at one place without changing each place in other files. 
* To make the logs file in order, each setting is seperated in different folders in `tmp`, which is demonstrated in configs. One can set up his/her own `tmp` hierarchy by changing the *flags* in configs. At the leaves layer of the hierarchy, each experiment is labeled with the timestamp when the code begins to run so that they won't have the same name and cover each other. Of course to distinguish between each other, one had better add a more concrete flag on the folder name after the timestamp.
* In `default configs.py`, one can specify different configs for different settings by using different class. The same for `env.py`.
* The `env.py`] won't be tracked or added or committed by git via `.gitignore`, and the reason is the data paths could be different on different machines. Besides, the path on a machine could leak privacy.

### `models`

This folder stores code that define models. It is suggested that models for different setting are places in different sub folders.

### `mylib`

This folder stores common functions / classes that could be used generally across different projects. Besides, the definitions of datset are also placed here. 

* `utils` stores common functions / classes that are leveraged by `models` and `scripts`
* `loss` stores various loss functions for classification, segmentation and so on.
* `evalution` stores various evalutaino metrics for classification, segmentation and so on, including *IOU*, *DICE*, etc.

You can add other common functions / classes in this folder based on particular tasks. For example, for 3D voxel tasks, there could be a `voxel transform.py`

### `scripts`

