# a-robust-training-scheme-to-defend-backdoor-attacks (RTS)
This is the implementation of the paper Regularization is the Key to Separate Backdoor Data.

For example, if you want to train a clean ResNet-18 model on the CIFAR-10 that is poisoned by the BadNets attack, where the poisoning rate is 0.1, you can directly run:

python scheme.py --dataset CIFAR10 --trigger_type gridTrigger --inject_portion 0.1
