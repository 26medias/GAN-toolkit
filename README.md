# GAN-toolkit

TF/Keras 2 GAN toolkit

## Install

pip install keras==2.2.4

## Usage
	
	import pyplot as plt
	
	from models import FaceTranslation
	model	= FaceTranslation()
	
	session = model.create_session()
	session.img_input(img)
	session.img_targets([img1, img2, ...])
	outputs = session.output()
	
	plt.imshow(result_img)

-----------------------------

## Credit

- https://github.com/shaoanlu/fewshot-face-translation-GAN
- https://github.com/shaoanlu/face_toolbox_keras