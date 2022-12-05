# ***⚡ [TmPL](): [T]()e[mpl]()ate for [P]()ytorch [L]()ightning***

<!--
# ***<font color=#0668E1>TmPL</font>: <font color=#0668E1>T</font>e<font color=#0668E1>mpl</font>ate for <font color=#0668E1>P</font>ytorch <font color=#0668E1>L</font>ightning***
-->

![](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
![](https://img.shields.io/badge/PyTorch-1.8%2B-red)
![](https://img.shields.io/badge/Lightning-1.8-blue)
![](https://img.shields.io/badge/Hydra-1.2-lightgrey)

[![](https://img.shields.io/github/license/npurson/tmpl)](LICENSE)
![](https://img.shields.io/badge/version-v0.2-blue)

[Docs](https://pytorch-lightning.readthedocs.io/en/stable/) &nbsp;•&nbsp;
[Installation](#installation) &nbsp;•&nbsp;
[Usage](#usage) &nbsp;•&nbsp;
[Reference](#reference) &nbsp;•&nbsp;
[Contributing](#contributing) &nbsp;•&nbsp;
[License](#license)

A template for rapid DL experimentation development, built upon [Lightning](https://lightning.ai/) and [Hydra](https://hydra.cc/) with best practice.

## What's New

***v0.2*** was released in 2022-12-01:

* Bumped to the latest Lightning v1.8
* Applies Hydra for configuring

## Installation

```
pip install -r requirements.txt
```

## Usage

1. **Training**

    ```shell
    python tools/train.py [--config-name config.yaml]
    ```

    * Overrides the default config file (the `config_name` specified in `hydra.main()`) with `--config-name`.
    * Override values in the loaded config from the command line with `+`.
    * Use `++` to override a config value if it's already in the config, or add it otherwise.
    * Refer to https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/ and https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more infomation.

2. **Further Development**

    1. Init tensors using `Tensor.to` and `register_buffer`

        ```python
        # before lightning
        def forward(self, x):
            z = torch.Tensor(2, 3)
            z = z.cuda(0)

        # with lightning
        def forward(self, x):
            z = torch.Tensor(2, 3)
            z = z.to(x)

        class LitModel(LightningModule):
        def __init__(self):
            ...
            self.register_buffer("sigma", torch.eye(3))
        ```

    2. Note if you use any built in metrics or custom metrics that use TorchMetrics, these do not need to be updated and are automatically handled for you.

    3. This Lightning implementation of DDP calls your script under the hood multiple times with the correct environment variables

        ```bash
        MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
        ```

## Reference

* https://pytorch-lightning.readthedocs.io/en/stable/
* https://hydra.cc/docs/intro/
* https://github.com/Lightning-AI/lightning

## Contributing

Contributions are always welcome and appreciated! \
Feel free to open issues/PRs! 🎉

## License

Released under the [MIT](LICENSE) License.
