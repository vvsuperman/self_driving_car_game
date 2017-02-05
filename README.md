# self_driving_car_game
could you pls to help me to improve my code ? And give me some advise to improve my action, the angle is always the same, thks:)

1. I use only the center image, First I preprocess the image: crop the image to 2/3,cut the top 1/3 useless image off, then resize the image to (200,66) which is applicable for the NVIDIA network(may I resize the image small, e.g, (100,33), if the NVIDIA can  accept the small image?), then I use normalize_grayscale to normalize the image.
2. use train_test_split to split 10% for test, then still use train_test_split to split 20% for validate
3. build the NVIDIA network, which as follow

Layer (type)                     Output Shape          Param #     Connected to                     

convolution2d_1 (Convolution2D)  (None, 33, 100, 24)   1824        convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 33, 100, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 50, 36)    21636       activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 17, 50, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 9, 25, 48)     43248       activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 9, 25, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 9, 25, 64)     27712       activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 9, 25, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 9, 25, 64)     36928       activation_4[0][0]               
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 9, 25, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 14400)         0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          16762764    flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           activation_6[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        activation_7[0][0]               
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         activation_8[0][0]               
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 10)            0           activation_9[0][0]               
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             11          dropout_2[0][0]                  

4. write a generator to generate the image, then use fit_generator to import the image, which have a 
5. use mse and adm to fit the dataset
6. modify the drive.py preprocess the image to apply the predict


but some wrong with my project:
the angle of my car is always 0.2, I use about 4K image,when I run my model.py, the console is like that:

113/4000 [..............................] - ETA: 1833s - loss: 0.0475 - acc: 0.7876
 114/4000 [..............................] - ETA: 1832s - loss: 0.0471 - acc: 0.7895
 115/4000 [..............................] - ETA: 1830s - loss: 0.0467 - acc: 0.7913

and the final test accuacy is about 0.7
then I use the model to run the car, but the angle is seems that it's always 0.2

-0.04029325023293495 0.2  
-0.04029325023293495 0.2  
-0.04029325023293495 0.2  
-0.04029325023293495 0.2  

could you pls to help me to improve my code ? And give me some advise to improve my action, thks:)











