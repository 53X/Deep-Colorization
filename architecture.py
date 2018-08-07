#library dependencies

from keras.layers import Conv2D,Input,Dense,Flatten
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model



def encoder():

	'''

	This is the encoder model in our entire architecture

	'''

	input_layer=Input(shape=(224,224,1),name='input')
	
	first_conv=Conv2D(filters=64,kernel_size=(3,3),padding='same',data_format='channels_last',
					  strides=(2,2),name='encoder_1',activation='relu')(input_layer)
	second_conv=Conv2D(filters=128,kernel_size=(3,3),padding='same',data_format='channels_last',
					   strides=(1,1),name='encoder_2',activation='relu')(first_conv)
	third_conv=Conv2D(filters=128,kernel_size=(3,3),padding='valid',data_format='channels_last',
					  strides=(2,2),name='encoder_3',activation='relu')(second_conv)
	fourth_conv=Conv2D(filters=256,kernel_size=(3,3),padding='same',data_format='channels_last',
					   strides=(1,1),name='encoder_4',activation='relu')(third_conv)
	fifth_conv=Conv2D(filters=256,kernel_size=(3,3),padding='same',data_format='channels_last',
					  strides=(2,2),name='encoder_5',activation='relu')(fourth_conv)
	sixth_conv=Conv2D(filters=512,kernel_size=(3,3),padding='valid',data_format='channels_last',
					  strides=(1,1),name='encoder_6',activation='relu')(fifth_conv)
	seventh_conv=Conv2D(filters=512,kernel_size=(3,3),padding='same',data_format='channels_last',
					    strides=(1,1),name='encoder_7',activation='relu')(sixth_conv)
	eighth_conv=Conv2D(filters=256,kernel_size=(3,3),padding='same',data_format='channels_last',
					   strides=(1,1),name='encoder_8',activation='tanh')(seventh_conv)


	encoder = Model(inputs=input_layer, outputs=eighth_conv)

	return encoder				   

#Feature extractor 

def feature_extractor():

	inception_model=InceptionResNetV2(weights='imagenet', include_top=True, input_shape=(229,229,3))
	extractor_output = Dense(1001)(inception_model.get_layer('avg_pool').output)
	feature_extractor=Model(inputs=inception_model.inputs,outputs=extractor_output)

	return feature_extractor





if __name__ =='__main__':

	feature_extractor()
#Fusion 

