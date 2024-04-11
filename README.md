# a-robust-training-scheme-to-defend-backdoor-attacks (RTS)

BadNets attack to RestNet-18 on CIFAR-10
python scheme.py --trigger_type gridTrigger --dataset CIFAR10

python scheme.py --trigger_type squareTrigger --dataset CIFAR10
python scheme.py --trigger_type fourCornerTrigger --dataset CIFAR10
python scheme.py --trigger_type randomPixelTrigger --dataset CIFAR10
python scheme.py --trigger_type signalTrigger --dataset CIFAR10     
python scheme.py --trigger_type trojanTrigger --dataset CIFAR10     
python scheme.py --trigger_type CLTrigger --dataset CIFAR10         
python scheme.py --trigger_type nashvilleTrigger --dataset CIFAR10   
python scheme.py --trigger_type onePixelTrigger --dataset CIFAR10      
python scheme.py --trigger_type wanetTrigger --dataset CIFAR10            
python scheme.py --trigger_type blendTrigger --dataset CIFAR10          
python scheme.py --trigger_type dynamicTrigger --dataset CIFAR10         
