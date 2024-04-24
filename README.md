# evidence_detection
Group_70_ED_A.ipynb is the main training file for the solution a (Transformer Encoder with Delta-GClip), and where the model is implemented. It produces a model for each epoch, such that each model can be evaluated. The files also contains some google.cloud functionality. Finally, there is code that downloads and extracts the zip for the GloVe embeddings. Downloaded from http://nlp.stanford.edu/data/glove.6B.zip.

Group_70_Prediction_A.ipynb is the file for the prediction for the solution a (Transformer Encoder with Delta-GClip), and also is demo code. It contains some google.cloud functionality for uploading files that may need to be removed depending on how you intend to run it. It takes the test and the model files from the same folder that it is in.

best_model_a.pth is the save of the model produced by Group_70_ED_A.ipynb, and used by Group_70_Prediction_A.ipynb.

ED_a is the model card for solution a (Transformer Encoder with Delta-GClip)

Group_70_a.csv contains the predictions for the test data.

Group_70_ED_B.ipynb is the main training file for the solution b (Deep learning-based approaches that do not employ transformer architectures), and where the model is implemented. It produces 'best_model_b.pth' which is the save of the best performing model. The files also contains some google.cloud functionality. Finally, there is code that downloads and extracts the zip for the GloVe embeddings. Downloaded from http://nlp.stanford.edu/data/glove.6B.zip.

Group70_Prediction_B.ipynb is the file for the prediction for the solution b (Deep learning-based approaches that do not employ transformer architectures), and also is demo code. It contains some google.cloud functionality for uploading files that may need to be removed depending on how you intend to run it. It takes the test and the model files from the same folder that it is in.

best_model_b.pth is the save of the model produced by Group_70_ED_B.ipynb, and used by Group70_Prediction_B.ipynb.

ED_b is the model card for solution b (Deep learning-based approaches that do not employ transformer architectures)

Group_70_b.csv contains the predictions for the test data.

ED_A uses encoder from https://github.com/d2l-ai/d2l-en, and GloVe embeddings from https://nlp.stanford.edu/projects/glove/.
