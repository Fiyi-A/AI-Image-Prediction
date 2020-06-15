from imageai.Prediction import ImagePrediction
import os

# get the current working directory for the execution path
execution_path=os.getcwd()

# instantiate the image prediction class
prediction = ImagePrediction()

# set the model to squeeze net
prediction.setModelTypeAsSqueezeNet()

# set the prediction model to the squeezenet algorithm
prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))

# load the prediction model
prediction.loadModel()

# change the "house.jpg" to any picture from the folder to predict the picture
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "house.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
