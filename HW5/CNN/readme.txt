## Training ##
''''''''''''''''''''''''''''''''''
	python CNN/CNN.py -test=False
''''''''''''''''''''''''''''''''''


## Testing ##
''''''''''''''''''''''''''''''''''
	python CNN/CNN.py 
''''''''''''''''''''''''''''''''''

p.s there are still other args for parameters setting.
	-savepath  where to savemodel
	-num_epoch  number of training epoch
	-lr  learning rate
	...
	You need to set to your own path!


## Result ##
Best result for this simple CNN model is only 74% in testing.

## Directory ##
---------------------
		|--CNN
		|	|--CNN.py
		|	|--model.py
		|	|--train.py
		|	|--test.py
		|
		|--model
		|	|--model.pt
		|
		|--hw5_data
			|--train
			|--test