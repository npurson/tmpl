# ***⚡ [TmPL](): [T]()e[mpl]()ate for [P]()ytorch [L]()ightning***

<!--
# ***<font color=#0668E1>TmPL</font>: <font color=#0668E1>T</font>e<font color=#0668E1>mpl</font>ate for <font color=#0668E1>P</font>ytorch <font color=#0668E1>L</font>ightning***
-->

![](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
![](https://img.shields.io/badge/PyTorch-1.8%2B-red)
![](https://img.shields.io/badge/Lightning-2.0-blue)
![](https://img.shields.io/badge/Hydra-1.3-lightgrey)

[![](https://img.shields.io/github/license/npurson/tmpl)](LICENSE)
![](https://img.shields.io/badge/version-v2.0rc0-blue)

[Docs](https://lightning.ai/docs/pytorch/stable/) &nbsp;•&nbsp;
[Installation](#installation) &nbsp;•&nbsp;
[Usage](#usage) &nbsp;•&nbsp;
[Reference](#reference) &nbsp;•&nbsp;
[Contributing](#contributing) &nbsp;•&nbsp;
[License](#license)

A template for rapid & flexible DL experimentation development, built upon [Lightning](https://lightning.ai/) & [Hydra](https://hydra.cc/) with best practice.

## What's New

***v2.0rc0*** released on Jul 12th, 2023:

* Upgraded to the latest Lightning v2.0+
    * NOTE: Compatibility issues may be introduced due to the changes in Lightning APIs, switch back to [v1.8](https://github.com/npurson/tmpl/tree/v1.8) if needed.

## Installation

```
pip install -r requirements.txt
```

* PyTorch and torchvision would be better manually installed first refered to https://pytorch.org/get-started/locally/.

## Usage

1. **Training**

    ```shell
    python tools/train.py [--config-name config[.yaml]] [datasets=cifar100] [trainer.devices=4] [data.loader.batch_size=16]
    ```

    * Override the default config file with `--config-name`.
    * You can also override any value in the loaded config from the command line, refer to the following for more infomation.
        * https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
        * https://hydra.cc/docs/advanced/hydra-command-line-flags/
        * https://hydra.cc/docs/advanced/override_grammar/basic/

2. **Tips for Further Development**

    1. Note if you use any built in metrics or custom metrics that use TorchMetrics, these do not need to be updated and are automatically handled for you.
    2. To be continued ...

## Reference

* https://pytorch-lightning.readthedocs.io/en/stable/
* https://hydra.cc/docs/intro/

## Contributing

Contributions are always welcome and appreciated! \
Feel free to open issues/PRs! 🎉

## License

Released under the [MIT](LICENSE) License.
