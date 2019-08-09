# HiveMind
My attempt to create a SC2 AI using Deepmind's pysc2 and ML techniques. This is not my first attempt using the SC2 APIs. I created a scripted AI as a project in University. That project lead me to the idea of using SC2 as a playground for using ML methods to play games.


# Machine Learning
After doing some brief research I decided the Reinforcement Learning was the path I would take. Since I was using Python it ultimately lead to me using Keras to make the model. I read over some tutorials (https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/ & https://elitedatascience.com/keras-tutorial-deep-learning-in-python) to get a better idea of what I was doing. 

# Deciding on the Action Set
I limited it to 3 thoughts for the action set to provide the model with. 
  1st:
      I thought giving it only 5-10 different actions to perform and to start learning with and then slowly develop a bigger action library. As someone who has played SC2 a bit, I realize for a beginning player learning to just play well with some units is better than trying to play with every single unit possible at the beginning.
    
  2nd:
    I thought that giving all of the actions to the model and let it decide, and then after it chooses check if the chosen action was valid.
  3rd:
    Somehow changing the action set every step so that it can choose from all of the available actions so it is making a valid move every step. 
