# plant-disease-swin
using swin to identify plant disease

## Model

The model used in this app is a Swin Transformer fine-tuned on the Plant Village dataset with added dropout for improved performance.

### Download the Model

Please download the model file from the following link and place it in the root directory of this repository:

[Download Model] https://drive.google.com/file/d/1Ao4HUSA3oRDegfGQgTFKs8gdAplWNGsl/view?usp=sharing

### Download the dataset

To download the dataset you need to have an API key (.json file download) from Kaggle. One of the first cells in the notebook will have you upload your json/kaggle file. 

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset 

## There was an error generating some of the attention maps after training was completed, I added another cell below the training cell to load the saved model state and re-create the attention maps. 

## Attribution

This code was developed and modified with the assistance of AI tools and includes modifications from various publicly available sources.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


To download the dataset you need to have an API key .json file from Kaggle. 

To run on streamlit from command prompt "streamlit run app2.py"

