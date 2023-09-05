# ***⚡ [TmPL](): [T]()e[mpl]()ate for [P]()ytorch [L]()ightning***

![](https://img.shields.io/badge/Python-3.8%2B-blue)
![](https://img.shields.io/badge/PyTorch-1.11%2B-red)
![](https://img.shields.io/badge/Lightning-2.0-blue)
![](https://img.shields.io/badge/Hydra-1.3-lightgrey)

[![](https://img.shields.io/github/license/npurson/tmpl)](LICENSE)
![](https://img.shields.io/badge/version-v2.0-blue)

[Lightning Docs](https://lightning.ai/docs/pytorch/stable/) &nbsp;•&nbsp;
[Installation](#installation) &nbsp;•&nbsp;
[Usage](#usage) &nbsp;•&nbsp;
[Reference](#reference) &nbsp;•&nbsp;
[Contributing](#contributing) &nbsp;•&nbsp;
[License](#license)

A template for rapid & flexible DL experimentation development, built upon [Lightning](https://lightning.ai/) & [Hydra](https://hydra.cc/) with best practice.

## What's New

***v2.0*** was released on Sep 5 '23.

## Installation

```
pip install -r requirements.txt
```

It is recommended to manually install PyTorch and Torchvision before running the installation command, referring to the official PyTorch website for [instructions](https://pytorch.org/get-started/locally/).

## Usage

0. **Setup**

    ```shell
    export PYTHONPATH=`pwd`:$PYTHONPATH
    ```

1. **Training**

    ```shell
    python tools/train.py [--config-name config[.yaml]] [trainer.devices=4] [data.loader.batch_size=16]
    ```

    * Override the default config file with `--config-name`.
    * You can also override any value in the loaded config from the command line, refer to the following for more infomation.
        * https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
        * https://hydra.cc/docs/advanced/hydra-command-line-flags/
        * https://hydra.cc/docs/advanced/override_grammar/basic/

2. **Tips for Further Development**

    The code is designed to be flexible and customizable to meet your specific needs. \
    Useful comments can be found in the source code.

## Reference

- [PyTorch Lightning Docs ↗](https://lightning.ai/docs/pytorch/stable/)
- [Hydra Docs ↗](https://hydra.cc/docs/intro/)

## Contributing

Contributions are welcome and appreciated! \
Feel free to open an issue or PR! 🎉

## License

Released under the [MIT](LICENSE) License.
