import os

files = os.listdir('./')

#print(files)

for file in files:
    if file.startswith('checkpoints-'):
        print(file)
    else:
        continue
    checkpoints = os.listdir(file)
    for checkpoint in checkpoints:
        if 'last.ckpt' not in checkpoint:
            #print('--',checkpoint)
            print('Removing',file+'/'+checkpoint)
            os.system(f'rm {file}/{checkpoint}')