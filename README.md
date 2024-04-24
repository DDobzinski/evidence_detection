# evidence_detection


Group_70_ED_B.ipynb is the main training file for the solution b (Deep learning-based approaches that do not employ transformer architectures), and where the model is implemented. It produces 'best_model_b.pth' which is the save of the best performing model. The files also contains some google.cloud functionality. Finally, there is code that downloads and extracts the zip for the GloVe embeddings. Downloaded from http://nlp.stanford.edu/data/glove.6B.zip.

Group70_Prediction_B.ipynb is the file for the prediction for the solution b (Deep learning-based approaches that do not employ transformer architectures), and also is demo code. It contains some google.cloud functionality for uploading files that may need to be removed depending on how you intend to run it. It takes the test and the model files from the same folder that it is in.

best_model_b.pth is the save of the model produced by Group_70_ED_B.ipynb, and used by Group70_Prediction_B.ipynb.

ED_b is the model card for solution b (Deep learning-based approaches that do not employ transformer architectures)

Group_70_b.csv contains the predictions for the test data.