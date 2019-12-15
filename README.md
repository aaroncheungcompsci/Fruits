# Identifying Fruits

This project demonstrates the knowledge gained from taking CSC 180: Intelligent Systems at California State University: Sacramento. 

In this project, we develop numerous convolutional neural networks (CNNs) in order to see if we can train and test one to a particular dataset. In this case, we have used the Fruits 360 dataset from kaggle.com, which consists of 80,000+ images of various types of fruits. A link to the dataset can be found [here](https://www.kaggle.com/moltean/fruits).

Due to time constraints of the semester, we have only decided to go with a portion of the dataset in order to reduce the train/test time so that we could finish our project in a timely manner (we used roughly 12,000 out of the 80,000 images). 

Throughout the notebook, you will see four different sections: Connecting with Google Drive, Preprocessing the data, Creating the CNNs, and Using Transfer Learning. Connecting with Google Drive was something that we had to research on our own, as we had to use Google Colab to make use of their GPU in order to train our neural networks appropriately. However, the other three sections essentially cover the majority of what was covered in the course.

## Preprocessing the data

In this section, we can observe the labelling of the data as well as going through each of the directories that had all the images from the dataset to import them into our notebook. The original dataset had already split the 80,000 images of fruits into the training and test sets (80% training and 20% test), so it was just a matter of selecting the same kind of fruits for both sets in order to preserve the train/test ratio.

## Creating the Convolutional Neural Networks (CNNs)

This section simply shows our process of developing our CNNs; it was a lot of trial and error on our part due to our limited knowledge of what is the best way to order the different layers and such.

### First CNN
The first CNN was taken from one of our previous projects in order to demonstrate what the lack of parameter tuning can do; obviously, the results were very poor. Based on the classification report after training the neural network, we can observe an accuracy of 26% and average F1 score of 0.10.

### Second CNN
Our second CNN followed what was supposedly an "ideal" way to order the different layers for the neural network. This "ideal" way was as follows:

- Conv2D 
- Max Pooling
- (Alternate as needed)
- Flatten
- Dense (as many as needed)
- Softmax

In addition to this, we only used the ReLU activation function for the layers that allowed us to add the type of activation function to be used for the layer. There was no particular reason as to why we did this, as much of what we experimented with was just trial and error. Given more time, however, we might have been able to do some more parameter tuning in order to see what might have worked the best. The classification report of this neural network shows that the accuracy and average F1 scores of the correct predictions were 93%.

### Third CNN
Finally, our third CNN was essentially an attempt to recreate the neural network that was proposed for the original dataset, provided by the same people who put together the dataset itself. The reason why we implemented this was curiosity; we wanted to see and understand why they built their model the way they did. A link to the academic paper containing the details of this network can be found [here](https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning), on page 18.

Not surprisingly, the neural network performed well, although interestingly it ended up doing just slightly worse than our second CNN (roughly 1-2% worse for the accuracy and average F1 score), which was simply using the recommended order of the layers for the neural networks. Our intuition says that this might be because we did not do any parameter tuning at all, which makes sense because all we did was simply observe how well our attempt of implementing their model did in comparison to the other two models we created.

## Using Transfer Learning
Transfer learning is a concept in which we take a model that was trained to do something similar to our problem at hand. It is worth noting and recalling that we only used a small fraction of the original dataset, and since transfer learning is something that is used when there is not much data to work with, we figured that we would like to see how well transfer learning performs in comparison to the CNNs created in the previous section.

We used the VGG16 model as our "transfer learning" neural network, and not surprisingly, it had 99% accuracy and average F1 score when running through the test set. We used the VGG16 model because it was something we used in class for a previous project where we had to make the model identify images correctly given a number of labels, and so we figured that this project in identifying fruits is pretty much the exact same problem.
