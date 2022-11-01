# xored_pets
Original code from https://keras.io/examples/vision/image_classification_from_scratch/
Set ~3000 Cats & 3000 Dogs pix, part of Kaggle Cats vs Dogs dataset
It was an idea to check training process on changed images
Original pixels were masked by nearest pixel (y+1, x+1) - XOR operation (xore.py file)
One pix set was XORed 100%, second - on half, by 50% RGB value of near pixel
Example - 

![](https://github.com/Repinoid/xored_pets/blob/main/maskedCat_Small.jpg?raw=true)

As shown below fully masked image set has some better result on ~8th epoch

![text](https://github.com/Repinoid/xored_pets/blob/main/Acc_Loss_Small.jpg?raw=true)

My PC has not GPU, so impossible to make more training with different parameters.
Maybe this idea with masking has some sense
