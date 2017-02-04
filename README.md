# self_driving_car_game
1. preprocess the image: crop the image to 2/3,cut the top 1/3 useless image off, then resize the image to (200,66) which is applicable for the NVIDIA network(may I resize the image small, e.g, (100,33), if the NVIDIA can  accept the small image?), then I use normalize_grayscale to normalize the image.

2. use train_test_split to split 10% for test, then still use train_test_split to split 20% for validate

3. build the NVIDIA network, which as follow