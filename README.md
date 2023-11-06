# Emotion-Based-Song-Suggestion-System-for-Tamil
An application that would provide a Tamil songs playlist based on your emotion.
* The dataset of Tamil songs has been manually curated and has over 800 songs belonging to 4 categories Sad, Happy, Angry, Calm.
* An Ensemble Learning Classifier is used to predict the emotion of the song based on features extracted such as Instrumentalness, Tempo, and etc. This classifier had an accuracy of 90%.
* For human Emotion Recognition the MobileNet model trained on FER2013 was used providing an accuracy of 75%.
* The user will have to take a photo from the webpage and then the emotion will be detected and a corresponding playlist will be generated.
* The components were integrated using Flask for backend and HTML for frontend.
