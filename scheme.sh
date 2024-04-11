python scheme.py --trigger_type squareTrigger --id 1 --gpuid 0              #
python scheme.py --trigger_type gridTrigger --id 1 --gpuid 0                #
python scheme.py --trigger_type fourCornerTrigger --id 1 --gpuid 0          #
python scheme.py --trigger_type randomPixelTrigger --id 1 --gpuid 0         #
python scheme.py --trigger_type signalTrigger --id 1 --gpuid 0              # SIG
python scheme.py --trigger_type trojanTrigger --id 1 --gpuid 1             # Trojan
python scheme.py --trigger_type CLTrigger --id 1 --gpuid 1                 # CL
python scheme.py --trigger_type nashvilleTrigger --id 1 --gpuid 1          # Nash
python scheme.py --trigger_type onePixelTrigger --id 1 --gpuid 1           # OnePixel
python scheme.py --trigger_type wanetTrigger --id 1 --gpuid 1              # WaNet
python scheme.py --trigger_type blendTrigger --id 1 --gpuid 1              # Blend
python scheme.py --trigger_type dynamicTrigger --id 1 --gpuid 1            # Dynamic


# python scheme.py --trigger_type dynamicTrigger    ×

# 'squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger', 'trojanTrigger', 'CLTrigger', 'dynamicTrigger', 'nashvilleTrigger', 'onePixelTrigger', 'wanetTrigger'


# 'gridTrigger'          BadNets
# 'fourCornerTrigger'    
# 'trojanTrigger'        Trojaning attack on Neural Networks
# 'blendTrigger'         Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning
# 'signalTrigger'        A New Backdoor Attack in Cnns by Training Set Corruption without Label Poisoning
# 'CLTrigger'            Label-Consistent Backdoor Attacks
# 'smoothTrigger'        Rethinking the backdoor attacks’ triggers: A frequency perspective
# 'dynamicTrigger'       Input-aware dynamic backdoor attack
# 'nashvilleTrigger'     Spectral signatures in backdoor attacks
# 'onePixelTrigger'      WaNet-Imperceptible Warping-based Backdoor Attack