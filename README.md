# ML tools 

This is a small custom made library to make experimentation faster when trying to build a good machine learning model.
It includes:
* custom transformers that are compatible with sklearn pipelines
* a data loading class to easily load train/test data and transform with various pipelines
* helper functions for the other scripts 


There are 4 kinds of pipelines you can use from the Pipes() class:

1. identity_pipe- transforms a pandas dataframe to a numpy array, no other transformations or preprocessing
2. base_pipe- transforms a pandas dataframe to a standardized numpy array
3. dummy_pipe- transforms a pandas dataframe to a standardized, one hot encoded numpy array
4. pca_sep_pipe- transforms a pandas dataframe to a standardized numpy array + PCA reduced one hot encodings as numpy
array

If desired, the Pipes() class has a flag use_pca that can be set to True to apply pca to pipes 
1, 2 , or 3. The default n_components is set to 10. You can set this variable as well when initializing the class like
so:
```python
pipe = Pipes(use_pca=True, reduce_to=20)
```

The DataPrep class has two methods: load_data() and transform()

* load_data() takes in train.csv and test.csv path and will generate
    1. training set: X_train, y_train
    2. validation set: X_val, y_val
    3. test set (from test data): test, test_id

* transform() takes the X_train, X_val, test arrays and transforms them all according to whichever pipeline has been set
to True. By default, if none of the flags are initialized as true, then the identity_pipe described above will be used.
