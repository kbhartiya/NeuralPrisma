import imageio
import os

filenames = sorted(os.listdir('./output'))

with imageio.get_writer('./GIFs/generated.gif',mode='I') as writer:
	for filename in filenames:
		image = imageio.imread(filename)
		writer.append_data(image)

		
