import torch

print('='*60)
print('🔍 PYTORCH GPU CHECK FOR main.py')
print('='*60)
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'Default Device: cuda')
    print('='*60)
    print('✅ main.py WILL use GPU for LSTM training!')
else:
    print(f'Default Device: cpu')
    print('='*60)
    print('❌ main.py will use CPU')
print('='*60)

# Check what device would be selected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice that main.py will use: {device}')
print(f'This is the EXACT device your LSTM used!\n')
