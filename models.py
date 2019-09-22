import cv2
import os
import numpy as np
from keras.layers import Input
from keras import backend as K

from utils import utils
from matplotlib import pyplot as plt

import networks.generator as gen

from verifier.face_verifier import FaceVerifier
from parser import face_parser
from detector import face_detector
from detector.iris_detector import IrisDetector

INPUT_SIZE = 224
LATENT_DIM = 512
NC_IN = 3

class FTSession:
	def __init__(self, ft_instance):
		self.ft	= ft_instance
	
	def img_input(self, img):
		self.src, self.mask, self.aligned_im, (self.x0, self.y0, self.x1, self.y1), self.landmarks = utils.get_src_inputs(img, self.ft.fd, self.ft.fp, self.ft.idet)
	
	def img_targets(self, imgs):
		self.tar, self.emb_tar = utils.get_tar_inputs(imgs, self.ft.fd, self.ft.fv)
	
	def output(self):
		self.out	= self.ft.inference(self.src, self.mask, self.tar, self.emb_tar)
		
		result_face	= np.squeeze(((self.out[0] + 1) * 255 / 2).astype(np.uint8))
		return self.out, result_face
		#return utils.post_process_result(self.src, self.ft.fd, result_face, self.aligned_im, self.src, self.x0, self.y0, self.x1, self.y1, self.landmarks)
		

class FaceTranslation:
	def __init__(self):
		self.input_size = INPUT_SIZE
		self.latent_dim = LATENT_DIM
		self.nc_in = NC_IN
		
		#Build the encoder & decoder
		self.encoder = self.build_encoder()
		self.decoder = self.build_decoder()
		try:
			self.encoder.load_weights("/content/face_translation/encoder.h5")
			self.decoder.load_weights("/content/face_translation/decoder.h5")
			print("Found checkpoints in weights folder. Built model with pre-trained weights.")
		except:
			print("Model built with default initializaiton.")
			pass
		
		image_size = (self.input_size, self.input_size, self.nc_in)
		inp_src = Input(shape=image_size)
		inp_tar = Input(shape=image_size)
		inp_segm = Input(shape=image_size)
		inp_emb = Input((self.latent_dim,))
		self.path_inference = K.function(
			[inp_src, inp_tar, inp_segm, inp_emb], 
			[self.decoder(self.encoder([inp_src, inp_tar, inp_segm]) + [inp_emb])]
		)
		
		self.init_models()
	
	def init_models(self):
		print("Initiating the face verifier...")
		self.fv = FaceVerifier(classes=512, weights_path="/content/face-recognition/facenet_keras_weights_VGGFace2.h5")
		
		print("Initiating the face parser...")
		self.fp = face_parser.FaceParser(path_bisenet_weights="/content/face-segmentation/BiSeNet_keras.h5")
		
		print("Initiating the face alignemnt...")
		self.fd = face_detector.FaceAlignmentDetector(fd_weights_path="/content/face-detector/s3fd_keras_weights.h5", lmd_weights_path="/content/face-alignment/2DFAN-4_keras.h5")
		
		print("Initiating the iris detector...")
		self.idet = IrisDetector(path_elg_weights="/content/eye-detector/elg_keras.h5")
	
	def load_weights(self, weights_path):
		self.encoder.load_weights(os.path.join(weights_path, "encoder.h5"))
		self.decoder.load_weights(os.path.join(weights_path, "decoder.h5"))
		
	def build_encoder(self):
		return gen.encoder(self.nc_in, self.input_size)
		
	def build_decoder(self):
		return gen.decoder(512, self.input_size//16, self.nc_in, self.latent_dim)
		
	def create_session(self):
		return FTSession(self)
		
	
	def preprocess_input(self, im):
		im = cv2.resize(im, (self.input_size, self.input_size))
		return im / 255 * 2 - 1
	
	def inference(self, src, mask, tar, emb_tar):
		return self.path_inference(			
			[
				self.preprocess_input(src)[None, ...], 
				self.preprocess_input(tar)[None, ...], 
				self.preprocess_input(mask.astype(np.uint8))[None, ...],
				emb_tar
			])
		