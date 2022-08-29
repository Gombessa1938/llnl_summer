# LLNL powergrid project

This repo has three sub projects, which are **Singular value decomposition analysis**, **Autoencoder**, **LSTM predictor** <br>


# SVD analysis
This notebook is trying to explore how fast the singular value decays and how does a low rank reconstruction represent the original data.


# Autoencoder

To run this project, first modify the three file address in ```dataset.py``` and run ```dataset.py```.<br>

```dataset.py``` should generated a ```concat.npy``` file in the same directory.

I included two pretrained weight for the autoencoder
```
model1_encoder.pt
model2_decoder.pt
```

After running ```dataset.py```, run ```inference_result.ipynb```. 

# LSTM predictor
This project is written in ```Keras```, I have found ```tf.keras``` is a lot faster in quick prototyping LSTM models. 
I included a pretrained weight in ```H5``` format. running the notebook should automatically load this weight.
```
trained_lstm
```

