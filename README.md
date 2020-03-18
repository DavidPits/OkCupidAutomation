# OkCupidAutomation
Attemping at automation of okcupid using Nerual networks with image processing & NLP.

Data - 
Swiping through the console using OkCupid interactive function will pass or like and save the pictures with the corresponding tags.
Augmented the data with "pretty faces" from various subreddit.
Still need to get less attractive data.

OkCupid- selenuim automation of sending msg , downloading picutres , swiping right\left , etc.
face_exctrations.py - extract faces only from picutres to avoid noise , using MTCNN , resizing all the pics to 150,150.
NNtraining - Training the nerual network to distinguish btw the taste of the users, hopefully distinct between facial features.
BioAnaylzer - Not completed yet , will try to use bag of words to see which words appear or not and give a score based on it , 
for example student , gym will be high. A possible impromvent is LSTM & W2V but I think it'll be an overkill.

NN used for image - added another dense layer on top of ResNet50 with ADAMS & learning rate of 0.01.

