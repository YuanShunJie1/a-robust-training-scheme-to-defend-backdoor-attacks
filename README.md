# a-robust-training-scheme-to-defend-backdoor-attacks (RTS)

This is the implementation of the paper **Regularization is the Key to Separate Backdoor Data**.

## Usage
 For example, if you want to train a clean ResNet-18 model on the CIFAR-10 that is poisoned by the BadNets attack, where the poisoning rate is 0.1, you can directly run:
```
python scheme.py --dataset CIFAR10 --trigger_type gridTrigger --inject_portion 0.1
```
More attacks:
```
python scheme.py --trigger_type squareTrigger --inject_portion 0.1
python scheme.py --trigger_type gridTrigger --inject_portion 0.1
python scheme.py --trigger_type fourCornerTrigger --inject_portion 0.1
python scheme.py --trigger_type randomPixelTrigger --inject_portion 0.1
python scheme.py --trigger_type signalTrigger --inject_portion 0.1
python scheme.py --trigger_type trojanTrigger --inject_portion 0.1
python scheme.py --trigger_type CLTrigger --inject_portion 0.1
python scheme.py --trigger_type nashvilleTrigger --inject_portion 0.1
python scheme.py --trigger_type onePixelTrigger --inject_portion 0.1
python scheme.py --trigger_type wanetTrigger --inject_portion 0.1
python scheme.py --trigger_type blendTrigger --inject_portion 0.1
python scheme.py --trigger_type dynamicTrigger --inject_portion 0.1
```
## Acknowledgements
Thanks to other researchers for their open-source codings, which helped us a lot.

DivideMix: https://github.com/LiJunnan1992/DivideMix

RNP: https://github.com/bboylyg/RNP

BackdoorBox: https://github.com/THUYimingLi/BackdoorBox

BackdoorBench: https://github.com/SCLBD/BackdoorBench

backdoor-toolbox: https://github.com/vtu81/backdoor-toolbox

