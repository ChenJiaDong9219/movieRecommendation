import os
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib


'''
this module includes two methods: download_data() and extract_data()

download_data() will download the movielen data to './ml-1m.zip'
extract_data() will extract files from .zip file
'''
def download_data():
	"""
	download movie data
	"""
	data_name = 'ml-1m'
	save_path = './ml-1m.zip'
	url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
#	hash_code = 'c4d9eecfca2ab87c1945afe126590906'
	if os.path.exists(save_path):
		print('{} is already exiting..'.format(data_name))
	else:
		with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(data_name)) as pbar:
			urlretrieve(url, save_path, pbar.hook)


def extract_data():
	"""
	extract data from ml-1m.zip
	"""
	data_name = 'ml-1m'
	data_path = './ml-1m.zip'
	extract_path = './'
	if not os.path.exists(extract_path):
		os.makedirs(extract_path)
		unzip(data_name, data_path, extract_path)
	print('extracting done')

def unzip(data_name, from_path, to_path):
	print('extracting {} ....'.format(data_name))
	with zipfile.ZipFile(from_path) as zf:
		zf.extractall(to_path)



class DLProgress(tqdm):
	"""
	Handle progress bar while downloading
	"""
	last_block = 0
	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		a hook function
		"""
		self.total=total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num



if __name__ == '__main__':
	download_data()
	extract_data()
