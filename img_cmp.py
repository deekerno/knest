from PIL import Image 
import imagehash
import os

def calc_hash(img_path):
	return imagehash.whash(Image.open(img_path))

def compare(hash1, hash2):
	return hash1 - hash2