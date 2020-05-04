# Self-Driving-Car-Using-Keras-and-Udacity-Simulator       
This repository contains files for training and testing a deep Neural Network aimed at behavioural Cloning in Car Driving using Udacity Simulator      

# Steps-To-Run  
1.First Step is to install all the dependencies on your machine.                    
                   
                   pip install -r requirements.txt                    
2.Download the udacity Simulator from their official github repo.                     
                   
                   https://github.com/udacity/self-driving-car-sim     
                                                                                    
3.In the third Step go ahead and start up the simulator in training mode and record Your training data and save it inside data folder.                      
                                                 
4.Now we are actually going to train our model on the data that you generated.                                   
                   
                   python model.py                                     
                  
In this model.py file you can play with some basic features like learning rate and batch_size or customize the model itself.  
                                                              
5.Now your best models should be saved after the training is finished so it's time to actually test it on the simulator.                          
                   
                   python drive.py your-best-model-name.h5                                            
  
Note - When you try to generate your custom data through the simulator you might come across a problem while running the model.py file.                               
       This is because your generated data_log.csv file is inaccurate and doesn't go hand in hand with the code. To fix the error, your generated data csv file must be modified to have it's first row as shown in the image below                           
                                                
![Fixing The Error](https://github.com/TarunTomar122/Self-Driving-Car-Using-Keras/blob/master/images/errorFix.png)                           
                                    
# Resources                                     
1.https://github.com/ManajitPal/BehavioralCloning                                 
2.https://github.com/udacity/self-driving-car-sim                             
3.https://github.com/udacity/CarND-Behavioral-Cloning-P3                                    
