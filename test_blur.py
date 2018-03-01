from PIL import Image
import os
import utils.blur as blur
import cv2


def test_blur(dir_path):
	for filename in os.listdir(dir_path):
		if not filename.startswith('.'):
			file = os.path.join(dir_path, filename)
			image = cv2.imread(file)
			greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			print(filename, ": Variance - ", blur.check_sharpness(greyscale, 100))
			print(filename, ": FFT Mean - ", blur.fft(greyscale))
			print(filename, ": Tenengrad - ", blur.teng(greyscale))
			print(filename, ": modLap - ", blur.lapm(greyscale))
			print(filename, ": sum coeffs - ", blur.sum_wave(greyscale))
			print("\n")


if __name__ == "__main__":
	test_blur('/users/ad/projects/knest/goo_test/')