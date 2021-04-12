from PIL import Image
import glob
import numpy as np
from matplotlib import cm


DATA_PATH="./hw2_data/task1,2_hybrid_pyramid"


if __name__=='__main__':
	for image_path in glob.glob(DATA_PATH+"/*"):
		image=Image.open(image_path)
		image=np.array(image)
		height,width,channel=image.shape
		output_image=Image.fromarray(np.uint8(image)).convert('RGB')
		output_image.show()


		# (-1)^(x+y) to center the transform
		for c in range(channel):
			for y in range(height):
				for x in range(width):
					image[y][x][c]*=(-1)**(x+y)

		# fourier transform
		F=np.fft.fft2(image)

		# H(u,v)
		H=np.zeros((height,width,channel))
		D0=30
		for c in range(channel):
			for v in range(height):
				for u in range(width):
					D=np.sqrt(u**2+v**2)
					H[v][u][c]=np.exp(-D**2/(2*D0))

		# F(u,v)*H(u,v)
		low_pass_image=F*H

		# Compute the inverse Fourier transformation
		F_inv=np.fft.ifft2(low_pass_image)

		# Obtain the real part 
		real_image=np.abs(F_inv)

		# multiply (-1)^(x+y)
		for c in range(channel):
			for y in range(height):
				for x in range(width):
					real_image[y][x][c]*=(-1)**(x+y)

		output_image=Image.fromarray(np.uint8(real_image)).convert('RGB')
		output_image.show()
		break