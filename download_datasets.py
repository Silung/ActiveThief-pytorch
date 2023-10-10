from tqdm import tqdm
import requests
import gzip, shutil
from zipfile import ZipFile
import os

########################################################
# Register for an account on the ImageNet website at   #
# http://image-net.org/signup.php?next=download-images #
########################################################

imagenet_username = 'Silung'
imagenet_password = input('Input Pwd')

########################################################

def download_file(url, path, session=None):
    # Based on https://stackoverflow.com/a/39217788/1825792
    # and https://stackoverflow.com/a/37573701/1825792

    full_path = os.path.join(dataset_dir, path)
    local_filename = os.path.join(full_path, url.split('/')[-1])

    if os.path.exists(local_filename):
        print(f'{local_filename} already existed!')
        return local_filename

    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    if session is None:
        r = requests.get(url, stream=True)
    else:
        r = session.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(local_filename, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)

    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

    return local_filename


dataset_dir = 'data'

########################################################

mnist = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ]

gtsr_training = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
gtsr_test     = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
gtsr_labels   = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'

# Supply login credentials for image-net.org
imagenet_login_payload = {'username': imagenet_username, 'password': imagenet_password}
imagenet_login_url = 'http://image-net.org/login?next=download-images'
imagenet_training = [
                        'https://image-net.org/data/downsample/Imagenet64_train_part1.zip',
                        'https://image-net.org/data/downsample/Imagenet64_train_part2.zip'
                    ]
imagenet_val = 'https://image-net.org/data/downsample/Imagenet64_val.zip'

########################################################

# MNIST
mnist_files = []
print('Downloading MNIST...')
for url in mnist:
    print(url)
    mnist_files.append(download_file(url, 'mnist'))
    print()
print()

print('Extracting MNIST...')
for gz_file in mnist_files:
    with gzip.open(gz_file, 'rb') as f_in:
        with open(gz_file[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_file)
print('DONE')
print()

########################################################

# GTSR
gtsr_files = []

print('Downloading GTSR...')
for url in [gtsr_training, gtsr_test, gtsr_labels]:
    print(url)
    gtsr_files.append(download_file(url, 'GTSRB'))
    print ()
print()

print('Extracting GTSR...')
for zip_file in gtsr_files:
    if zip_file.endswith('_GT.zip'):
        ZipFile(zip_file).extractall('/'.join(zip_file.split('/')[:-1]) + '/.')
    else:
        ZipFile(zip_file).extractall('/'.join(zip_file.split('/')[:-2]) + '/.')
    os.remove(zip_file)
print('DONE')
print()

########################################################

# ImageNet
print('Downloading ImageNet...')
imagenet_files = []
with requests.Session() as session:
    post = session.post(imagenet_login_url, data=imagenet_login_payload)

    for url in imagenet_training + [imagenet_val]:
        print(url)
        imagenet_files.append(download_file(url, 'Imagenet64', session))
        print()
print()

print('Preprocessing ImageNet...')
for zip_file in imagenet_files:
    print(zip_file)
    ZipFile(zip_file).extractall('/'.join(zip_file.split('/')[:-1]) + '/.')
    os.remove(zip_file)
print('DONE')
print()

