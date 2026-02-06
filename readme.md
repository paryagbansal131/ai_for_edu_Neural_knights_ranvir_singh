we first reduced the fps of the videos to reduce the scope of data to work with. then we isolated the face area to maximize efficiency. then we regularized the resolution of all the frames so model can work with every single one of em. we then passed the frame by frame data of each video to the model. the weights are in the .h5 file



dataset_2fpa - 2frames per second videos

frames 100x100 - fradesresized files all here(folder_


files
dataset  frameless.py - decrasing the fraes from 30 to 2

dataset/classify.py - for classification of label0 and label1

dataset/model.py - model 

dataset/task2.py - direct classification of task 2 into 4 levels

dataset/test.py - for basic face detection squares for framing

dataset/uniform_resolution_images.py - 100x100 images working

