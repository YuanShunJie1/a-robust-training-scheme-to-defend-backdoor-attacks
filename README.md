# a-robust-training-scheme-to-defend-backdoor-attacks
# BadNets attack to RestNet-18 on CIFAR-10
python scheme.py --trigger_type gridTrigger --id 1 --gpuid 0  

python scheme.py --trigger_type squareTrigger --id 1 --gpuid 0 
python scheme.py --trigger_type fourCornerTrigger --id 1 --gpuid 0    
python scheme.py --trigger_type randomPixelTrigger --id 1 --gpuid 0 
python scheme.py --trigger_type signalTrigger --id 1 --gpuid 0       
python scheme.py --trigger_type trojanTrigger --id 1 --gpuid 1       
python scheme.py --trigger_type CLTrigger --id 1 --gpuid 1              
python scheme.py --trigger_type nashvilleTrigger --id 1 --gpuid 1       
python scheme.py --trigger_type onePixelTrigger --id 1 --gpuid 1          
python scheme.py --trigger_type wanetTrigger --id 1 --gpuid 1             
python scheme.py --trigger_type blendTrigger --id 1 --gpuid 1             
python scheme.py --trigger_type dynamicTrigger --id 1 --gpuid 1           
