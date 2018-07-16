# UdacityImageClassification
# See <dlnd_image_classification.ipynb> for complete output 
# Below is only demo
<div tabindex="-1" id="notebook" class="border-box-sizing">

<div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

# Image Classification[¶](#Image-Classification)

In this project, you'll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset consists of airplanes, dogs, cats, and other objects. You'll preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded. You'll get to apply what you learned and build a convolutional, max pooling, dropout, and fully connected layers. At the end, you'll get to see your neural network's predictions on the sample images.

## Get the Data[¶](#Get-the-Data)

Run the following cell to download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [1]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="kn">from</span> <span class="nn">urllib.request</span> <span class="k">import</span> <span class="n">urlretrieve</span>
<span class="kn">from</span> <span class="nn">os.path</span> <span class="k">import</span> <span class="n">isfile</span><span class="p">,</span> <span class="n">isdir</span>
<span class="kn">from</span> <span class="nn">tqdm</span> <span class="k">import</span> <span class="n">tqdm</span>
<span class="kn">import</span> <span class="nn">problem_unittests</span> <span class="k">as</span> <span class="nn">tests</span>
<span class="kn">import</span> <span class="nn">tarfile</span>

<span class="n">cifar10_dataset_folder_path</span> <span class="o">=</span> <span class="s1">'cifar-10-batches-py'</span>

<span class="k">class</span> <span class="nc">DLProgress</span><span class="p">(</span><span class="n">tqdm</span><span class="p">):</span>
    <span class="n">last_block</span> <span class="o">=</span> <span class="mi">0</span>

    <span class="k">def</span> <span class="nf">hook</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">block_num</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">block_size</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">total_size</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">total</span> <span class="o">=</span> <span class="n">total_size</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">update</span><span class="p">((</span><span class="n">block_num</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">last_block</span><span class="p">)</span> <span class="o">*</span> <span class="n">block_size</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">last_block</span> <span class="o">=</span> <span class="n">block_num</span>

<span class="k">if</span> <span class="ow">not</span> <span class="n">isfile</span><span class="p">(</span><span class="s1">'cifar-10-python.tar.gz'</span><span class="p">):</span>
    <span class="k">with</span> <span class="n">DLProgress</span><span class="p">(</span><span class="n">unit</span><span class="o">=</span><span class="s1">'B'</span><span class="p">,</span> <span class="n">unit_scale</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">miniters</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">desc</span><span class="o">=</span><span class="s1">'CIFAR-10 Dataset'</span><span class="p">)</span> <span class="k">as</span> <span class="n">pbar</span><span class="p">:</span>
        <span class="n">urlretrieve</span><span class="p">(</span>
            <span class="s1">'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'</span><span class="p">,</span>
            <span class="s1">'cifar-10-python.tar.gz'</span><span class="p">,</span>
            <span class="n">pbar</span><span class="o">.</span><span class="n">hook</span><span class="p">)</span>

<span class="k">if</span> <span class="ow">not</span> <span class="n">isdir</span><span class="p">(</span><span class="n">cifar10_dataset_folder_path</span><span class="p">):</span>
    <span class="k">with</span> <span class="n">tarfile</span><span class="o">.</span><span class="n">open</span><span class="p">(</span><span class="s1">'cifar-10-python.tar.gz'</span><span class="p">)</span> <span class="k">as</span> <span class="n">tar</span><span class="p">:</span>
        <span class="n">tar</span><span class="o">.</span><span class="n">extractall</span><span class="p">()</span>
        <span class="n">tar</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>

<span class="n">tests</span><span class="o">.</span><span class="n">test_folder_path</span><span class="p">(</span><span class="n">cifar10_dataset_folder_path</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stderr output_text">

<pre>/home/abhiyantaabhishek1/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
</pre>

</div>

</div>

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>All files found!
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

## Explore the Data[¶](#Explore-the-Data)

The dataset is broken into batches to prevent your machine from running out of memory. The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:

*   airplane
*   automobile
*   bird
*   cat
*   deer
*   dog
*   frog
*   horse
*   ship
*   truck

Understanding a dataset is part of making predictions on the data. Play around with the code cell below by changing the `batch_id` and `sample_id`. The `batch_id` is the id for a batch (1-5). The `sample_id` is the id for a image and label pair in the batch.

Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are the labels in order or random?". Answers to questions like these will help you preprocess the data and end up with better predictions.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [2]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="o">%</span><span class="k">matplotlib</span> inline
<span class="o">%</span><span class="k">config</span> InlineBackend.figure_format = 'retina'

<span class="kn">import</span> <span class="nn">helper</span> 
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>

<span class="c1"># Explore the dataset</span>
<span class="n">batch_id</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">sample_id</span> <span class="o">=</span> <span class="mi">3</span>
<span class="n">helper</span><span class="o">.</span><span class="n">display_stats</span><span class="p">(</span><span class="n">cifar10_dataset_folder_path</span><span class="p">,</span> <span class="n">batch_id</span><span class="p">,</span> <span class="n">sample_id</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Stats of batch 1:
Samples: 10000
Label Counts: {0: 1005, 1: 974, 2: 1032, 3: 1016, 4: 999, 5: 937, 6: 1030, 7: 1001, 8: 1025, 9: 981}
First 20 Labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7, 2, 9, 9, 9, 3, 2, 6]

Example of Image 3:
Image - Min Value: 4 Max Value: 234
Image - Shape: (32, 32, 3)
Label - Label Id: 4 Name: deer
</pre>

</div>

</div>

<div class="output_area">



</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

## Implement Preprocess Functions[¶](#Implement-Preprocess-Functions)

### Normalize[¶](#Normalize)

In the cell below, implement the `normalize` function to take in image data, `x`, and return it as a normalized Numpy array. The values should be in the range of 0 to 1, inclusive. The return object should be the same shape as `x`.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [3]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">normalize</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Normalize a list of sample image data in the range of 0 to 1</span>
 <span class="sd">: x: List of image data.  The image shape is (32, 32, 3)</span>
 <span class="sd">: return: Numpy array of normalize data</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>

    <span class="k">if</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">-</span><span class="n">np</span><span class="o">.</span><span class="n">min</span><span class="p">(</span><span class="n">x</span><span class="p">))</span><span class="o">!=</span><span class="mi">0</span><span class="p">:</span>
        <span class="n">normalized</span> <span class="o">=</span> <span class="p">(</span><span class="n">x</span><span class="o">-</span><span class="n">np</span><span class="o">.</span><span class="n">min</span><span class="p">(</span><span class="n">x</span><span class="p">))</span><span class="o">/</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">x</span><span class="p">)</span><span class="o">-</span><span class="n">np</span><span class="o">.</span><span class="n">min</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">normalized</span> <span class="o">=</span> <span class="mi">1</span><span class="o">/</span><span class="p">(</span><span class="n">x</span> <span class="p">)</span>
    <span class="k">return</span> <span class="n">normalized</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_normalize</span><span class="p">(</span><span class="n">normalize</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Tests Passed
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### One-hot encode[¶](#One-hot-encode)

Just like the previous code cell, you'll be implementing a function for preprocessing. This time, you'll implement the `one_hot_encode` function. The input, `x`, are a list of labels. Implement the function to return the list of labels as One-Hot encoded Numpy array. The possible values for labels are 0 to 9\. The one-hot encoding function should return the same encoding for each value between each call to `one_hot_encode`. Make sure to save the map of encodings outside the function.

**Hint:**

Look into LabelBinarizer in the preprocessing module of sklearn.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [4]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">one_hot_encode</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>

    <span class="n">eccodemap</span><span class="o">=</span><span class="p">[]</span>

    <span class="k">for</span> <span class="n">val</span> <span class="ow">in</span> <span class="n">x</span><span class="p">:</span>
        <span class="n">temp1</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="n">dtype</span><span class="o">=</span><span class="nb">int</span><span class="p">)</span>
        <span class="n">temp1</span><span class="p">[</span><span class="n">val</span><span class="p">]</span><span class="o">=</span><span class="mi">1</span>
        <span class="n">eccodemap</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">temp1</span><span class="p">)</span>

    <span class="sd">"""</span>
 <span class="sd">One hot encode a list of sample labels. Return a one-hot encoded vector for each label.</span>
 <span class="sd">: x: List of sample Labels</span>
 <span class="sd">: return: Numpy array of one-hot encoded labels</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">eccodemap</span><span class="p">)</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_one_hot_encode</span><span class="p">(</span><span class="n">one_hot_encode</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Tests Passed
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Randomize Data[¶](#Randomize-Data)

As you saw from exploring the data above, the order of the samples are randomized. It doesn't hurt to randomize it again, but you don't need to for this dataset.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

## Preprocess all the data and save it[¶](#Preprocess-all-the-data-and-save-it)

Running the code cell below will preprocess all the CIFAR-10 data and save it to file. The code below also uses 10% of the training data for validation.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [5]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL</span>
<span class="sd">"""</span>
<span class="c1"># Preprocess Training, Validation, and Testing Data</span>
<span class="n">helper</span><span class="o">.</span><span class="n">preprocess_and_save_data</span><span class="p">(</span><span class="n">cifar10_dataset_folder_path</span><span class="p">,</span> <span class="n">normalize</span><span class="p">,</span> <span class="n">one_hot_encode</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

# Check Point[¶](#Check-Point)

This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [6]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL</span>
<span class="sd">"""</span>
<span class="kn">import</span> <span class="nn">pickle</span>
<span class="kn">import</span> <span class="nn">problem_unittests</span> <span class="k">as</span> <span class="nn">tests</span>
<span class="kn">import</span> <span class="nn">helper</span>
<span class="kn">import</span> <span class="nn">problem_unittests</span> <span class="k">as</span> <span class="nn">tests</span>
<span class="c1"># Load the Preprocessed Validation data</span>
<span class="n">valid_features</span><span class="p">,</span> <span class="n">valid_labels</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="nb">open</span><span class="p">(</span><span class="s1">'preprocess_validation.p'</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">'rb'</span><span class="p">))</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

## Build the network[¶](#Build-the-network)

For the neural network, you'll build each layer into a function. Most of the code you've seen has been outside of functions. To test your code more thoroughly, we require that you put each layer in a function. This allows us to give you better feedback and test for simple mistakes using our unittests before you submit your project.

> **Note:** If you're finding it hard to dedicate enough time for this course each week, we've provided a small shortcut to this part of the project. In the next couple of problems, you'll have the option to use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages to build each layer, except the layers you build in the "Convolutional and Max Pooling Layer" section. TF Layers is similar to Keras's and TFLearn's abstraction to layers, so it's easy to pickup.
> 
> However, if you would like to get the most out of this course, try to solve all the problems _without_ using anything from the TF Layers packages. You **can** still use classes from other packages that happen to have the same name as ones you find in TF Layers! For example, instead of using the TF Layers version of the `conv2d` class, [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d), you would want to use the TF Neural Network version of `conv2d`, [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d).

Let's begin!

### Input[¶](#Input)

The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. Implement the following functions

*   Implement `neural_net_image_input`
    *   Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
    *   Set the shape using `image_shape` with batch size set to `None`.
    *   Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
*   Implement `neural_net_label_input`
    *   Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
    *   Set the shape using `n_classes` with batch size set to `None`.
    *   Name the TensorFlow placeholder "y" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
*   Implement `neural_net_keep_prob_input`
    *   Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) for dropout keep probability.
    *   Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).

These names will be used at the end of the project to load your saved model.

Note: `None` for shapes in TensorFlow allow for a dynamic size.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [7]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">tensorflow</span> <span class="k">as</span> <span class="nn">tf</span>

<span class="k">def</span> <span class="nf">neural_net_image_input</span><span class="p">(</span><span class="n">image_shape</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Return a Tensor for a batch of image input</span>
 <span class="sd">: image_shape: Shape of the images</span>
 <span class="sd">: return: Tensor for image input.</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>
    <span class="k">return</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">,</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="kc">None</span><span class="p">,</span><span class="n">image_shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span><span class="n">image_shape</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="n">image_shape</span><span class="p">[</span><span class="mi">2</span><span class="p">]),</span><span class="n">name</span><span class="o">=</span><span class="s2">"x"</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">neural_net_label_input</span><span class="p">(</span><span class="n">n_classes</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Return a Tensor for a batch of label input</span>
 <span class="sd">: n_classes: Number of classes</span>
 <span class="sd">: return: Tensor for label input.</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>
    <span class="k">return</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">,</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="kc">None</span><span class="p">,</span><span class="n">n_classes</span><span class="p">),</span><span class="n">name</span><span class="o">=</span><span class="s2">"y"</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">neural_net_keep_prob_input</span><span class="p">():</span>
    <span class="sd">"""</span>
 <span class="sd">Return a Tensor for keep probability</span>
 <span class="sd">: return: Tensor for keep probability.</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>
    <span class="k">return</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">,</span><span class="n">name</span><span class="o">=</span><span class="s2">"keep_prob"</span><span class="p">)</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tf</span><span class="o">.</span><span class="n">reset_default_graph</span><span class="p">()</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_nn_image_inputs</span><span class="p">(</span><span class="n">neural_net_image_input</span><span class="p">)</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_nn_label_inputs</span><span class="p">(</span><span class="n">neural_net_label_input</span><span class="p">)</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_nn_keep_prob_inputs</span><span class="p">(</span><span class="n">neural_net_keep_prob_input</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Image Input Tests Passed.
Label Input Tests Passed.
Keep Prob Tests Passed.
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Convolution and Max Pooling Layer[¶](#Convolution-and-Max-Pooling-Layer)

Convolution layers have a lot of success with images. For this code cell, you should implement the function `conv2d_maxpool` to apply convolution then max pooling:

*   Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
*   Apply a convolution to `x_tensor` using weight and `conv_strides`.
    *   We recommend you use same padding, but you're welcome to use any padding.
*   Add bias
*   Add a nonlinear activation to the convolution.
*   Apply Max Pooling using `pool_ksize` and `pool_strides`.
    *   We recommend you use same padding, but you're welcome to use any padding.

**Note:** You **can't** use [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) for **this** layer, but you can still use TensorFlow's [Neural Network](https://www.tensorflow.org/api_docs/python/tf/nn) package. You may still use the shortcut option for all the **other** layers.

**Hint:**

When unpacking values as an argument in Python, look into the [unpacking](https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists) operator.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [8]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">conv2d_maxpool</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="n">conv_num_outputs</span><span class="p">,</span> <span class="n">conv_ksize</span><span class="p">,</span> <span class="n">conv_strides</span><span class="p">,</span> <span class="n">pool_ksize</span><span class="p">,</span> <span class="n">pool_strides</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Apply convolution then max pooling to x_tensor</span>
 <span class="sd">:param x_tensor: TensorFlow Tensor</span>
 <span class="sd">:param conv_num_outputs: Number of outputs for the convolutional layer</span>
 <span class="sd">:param conv_ksize: kernal size 2-D Tuple for the convolutional layer</span>
 <span class="sd">:param conv_strides: Stride 2-D Tuple for convolution</span>
 <span class="sd">:param pool_ksize: kernal size 2-D Tuple for pool</span>
 <span class="sd">:param pool_strides: Stride 2-D Tuple for pool</span>
 <span class="sd">: return: A tensor that represents convolution and max pooling of x_tensor</span>
 <span class="sd">"""</span>

    <span class="c1"># TODO: Implement Function</span>
    <span class="n">weight</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">([</span><span class="n">conv_ksize</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">conv_ksize</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">x_tensor</span><span class="o">.</span><span class="n">get_shape</span><span class="p">()</span><span class="o">.</span><span class="n">as_list</span><span class="p">()[</span><span class="mi">3</span><span class="p">],</span> <span class="n">conv_num_outputs</span><span class="p">],</span><span class="n">mean</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span><span class="n">stddev</span><span class="o">=</span><span class="mf">0.1</span><span class="p">))</span>

    <span class="n">bias</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">conv_num_outputs</span><span class="p">))</span>
    <span class="n">conv_layer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="n">weight</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="n">conv_strides</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">conv_strides</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'SAME'</span><span class="p">)</span>
    <span class="n">conv_layer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">bias_add</span><span class="p">(</span><span class="n">conv_layer</span><span class="p">,</span> <span class="n">bias</span><span class="p">)</span>
    <span class="n">conv_layer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">conv_layer</span><span class="p">)</span>
    <span class="n">conv_layer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">max_pool</span><span class="p">(</span><span class="n">conv_layer</span><span class="p">,</span> <span class="n">ksize</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="n">pool_ksize</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">pool_ksize</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="mi">1</span><span class="p">],</span>
                                <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="n">pool_strides</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">pool_strides</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'SAME'</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">conv_layer</span> 

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_con_pool</span><span class="p">(</span><span class="n">conv2d_maxpool</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Tests Passed
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Flatten Layer[¶](#Flatten-Layer)

Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor. The output should be the shape (_Batch Size_, _Flattened Image Size_). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [9]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">flatten</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Flatten x_tensor to (Batch Size, Flattened Image Size)</span>
 <span class="sd">: x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.</span>
 <span class="sd">: return: A tensor of size (Batch Size, Flattened Image Size).</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>
    <span class="k">return</span> <span class="n">tf</span><span class="o">.</span><span class="n">layers</span><span class="o">.</span><span class="n">flatten</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">)</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_flatten</span><span class="p">(</span><span class="n">flatten</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Tests Passed
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Fully-Connected Layer[¶](#Fully-Connected-Layer)

Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (_Batch Size_, _num_outputs_). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [10]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">fully_conn</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="n">num_outputs</span><span class="p">):</span>

    <span class="c1"># TODO: Implement Function</span>
    <span class="n">shape</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">x_tensor</span><span class="o">.</span><span class="n">get_shape</span><span class="p">()</span><span class="o">.</span><span class="n">as_list</span><span class="p">()[</span><span class="mi">1</span><span class="p">:])</span><span class="o">.</span><span class="n">prod</span><span class="p">()</span>

    <span class="n">weights</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">([</span><span class="n">shape</span><span class="p">,</span> <span class="n">num_outputs</span><span class="p">],</span><span class="n">mean</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">stddev</span><span class="o">=</span><span class="mf">0.1</span><span class="p">))</span>
    <span class="n">bias</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">([</span><span class="n">num_outputs</span><span class="p">]))</span>

    <span class="c1"># Fully convolution layer.</span>

    <span class="k">return</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">bias_add</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="n">weights</span><span class="p">),</span> <span class="n">bias</span><span class="p">))</span> 
    <span class="c1">#return tf.contrib.layers.fully_connected(x_tensor, num_outputs, tf.nn.relu)</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_fully_conn</span><span class="p">(</span><span class="n">fully_conn</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Tests Passed
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Output Layer[¶](#Output-Layer)

Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (_Batch Size_, _num_outputs_). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

**Note:** Activation, softmax, or cross entropy should **not** be applied to this.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [11]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">output</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="n">num_outputs</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Apply a output layer to x_tensor using weight and bias</span>
 <span class="sd">: x_tensor: A 2-D tensor where the first dimension is batch size.</span>
 <span class="sd">: num_outputs: The number of output that the new tensor should be.</span>
 <span class="sd">: return: A 2-D tensor where the second dimension is num_outputs.</span>
 <span class="sd">"""</span>
    <span class="n">shape</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">x_tensor</span><span class="o">.</span><span class="n">get_shape</span><span class="p">()</span><span class="o">.</span><span class="n">as_list</span><span class="p">()[</span><span class="mi">1</span><span class="p">:])</span><span class="o">.</span><span class="n">prod</span><span class="p">()</span>

    <span class="n">weights</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">([</span><span class="n">shape</span><span class="p">,</span> <span class="n">num_outputs</span><span class="p">],</span><span class="n">mean</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">stddev</span><span class="o">=</span><span class="mf">0.1</span><span class="p">))</span>
    <span class="n">bias</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">([</span><span class="n">num_outputs</span><span class="p">]))</span>

    <span class="k">return</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">bias_add</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="n">weights</span><span class="p">),</span> <span class="n">bias</span><span class="p">)</span>
    <span class="c1">#tf.contrib.layers.fully_connected(x_tensor, num_outputs)</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_output</span><span class="p">(</span><span class="n">output</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Tests Passed
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Create Convolutional Model[¶](#Create-Convolutional-Model)

Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits. Use the layers you created above to create this model:

*   Apply 1, 2, or 3 Convolution and Max Pool layers
*   Apply a Flatten Layer
*   Apply 1, 2, or 3 Fully Connected Layers
*   Apply an Output Layer
*   Return the output
*   Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [12]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">conv_net</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">):</span>

  <span class="c1"># TODO: Apply 1, 2, or 3 Convolution and Max Pool layers</span>
  <span class="c1">#    Play around with different number of outputs, kernel size and stride</span>
  <span class="c1"># Function Definition from Above:</span>
  <span class="c1">#    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)</span>
  <span class="n">conv_num_outputs</span><span class="o">=</span><span class="mi">64</span>
  <span class="n">conv_ksize</span><span class="o">=</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span><span class="mi">3</span><span class="p">)</span>
  <span class="n">conv_strides</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span>
  <span class="n">pool_ksize</span><span class="o">=</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
  <span class="n">pool_strides</span><span class="o">=</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">conv2d_maxpool</span><span class="p">(</span><span class="n">x</span><span class="p">,</span><span class="mi">32</span><span class="p">,</span> <span class="n">conv_ksize</span><span class="p">,</span> <span class="n">conv_strides</span><span class="p">,</span> <span class="n">pool_ksize</span><span class="p">,</span> <span class="n">pool_strides</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">conv2d_maxpool</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span><span class="mi">64</span><span class="p">,</span> <span class="n">conv_ksize</span><span class="p">,</span> <span class="n">conv_strides</span><span class="p">,</span> <span class="n">pool_ksize</span><span class="p">,</span> <span class="n">pool_strides</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">conv2d_maxpool</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span><span class="mi">128</span><span class="p">,</span> <span class="n">conv_ksize</span><span class="p">,</span> <span class="n">conv_strides</span><span class="p">,</span> <span class="n">pool_ksize</span><span class="p">,</span> <span class="n">pool_strides</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">conv2d_maxpool</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span><span class="mi">256</span><span class="p">,</span> <span class="n">conv_ksize</span><span class="p">,</span> <span class="n">conv_strides</span><span class="p">,</span> <span class="n">pool_ksize</span><span class="p">,</span> <span class="n">pool_strides</span><span class="p">)</span>
  <span class="c1"># TODO: Apply a Flatten Layer</span>
  <span class="c1"># Function Definition from Above:</span>
  <span class="c1">#   flatten(x_tensor)</span>
  <span class="c1">#x_tensor=tf.nn.dropout(x_tensor,keep_prob)</span>
  <span class="n">x_tensor</span> <span class="o">=</span> <span class="n">flatten</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">)</span>

  <span class="c1"># TODO: Apply 1, 2, or 3 Fully Connected Layers</span>
  <span class="c1">#    Play around with different number of outputs</span>
  <span class="c1"># Function Definition from Above:</span>
  <span class="c1">#   fully_conn(x_tensor, num_outputs)</span>
  <span class="n">num_outputs</span><span class="o">=</span><span class="mi">10</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">fully_conn</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="mi">1024</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span><span class="n">keep_prob</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">fully_conn</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="mi">512</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span><span class="n">keep_prob</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">fully_conn</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="mi">256</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span><span class="n">keep_prob</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">fully_conn</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="mi">128</span><span class="p">)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span><span class="n">keep_prob</span><span class="p">)</span>

  <span class="c1"># TODO: Apply an Output Layer</span>
  <span class="c1">#    Set this to the number of classes</span>
  <span class="c1"># Function Definition from Above:</span>
  <span class="c1">#   output(x_tensor, num_outputs)</span>
  <span class="n">x_tensor</span><span class="o">=</span><span class="n">output</span><span class="p">(</span><span class="n">x_tensor</span><span class="p">,</span> <span class="n">num_outputs</span><span class="p">)</span>

  <span class="c1"># TODO: return output</span>
  <span class="k">return</span> <span class="n">x_tensor</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>

<span class="c1">##############################</span>
<span class="c1">## Build the Neural Network ##</span>
<span class="c1">##############################</span>

<span class="c1"># Remove previous weights, bias, inputs, etc..</span>
<span class="n">tf</span><span class="o">.</span><span class="n">reset_default_graph</span><span class="p">()</span>

<span class="c1"># Inputs</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">neural_net_image_input</span><span class="p">((</span><span class="mi">32</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">3</span><span class="p">))</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">neural_net_label_input</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
<span class="n">keep_prob</span> <span class="o">=</span> <span class="n">neural_net_keep_prob_input</span><span class="p">()</span>

<span class="c1"># Model</span>
<span class="n">logits</span> <span class="o">=</span> <span class="n">conv_net</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">)</span>

<span class="c1"># Name logits Tensor, so that is can be loaded from disk after training</span>
<span class="n">logits</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">identity</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="s1">'logits'</span><span class="p">)</span>

<span class="c1"># Loss and Optimizer</span>
<span class="n">cost</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">reduce_mean</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">softmax_cross_entropy_with_logits</span><span class="p">(</span><span class="n">logits</span><span class="o">=</span><span class="n">logits</span><span class="p">,</span> <span class="n">labels</span><span class="o">=</span><span class="n">y</span><span class="p">))</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">AdamOptimizer</span><span class="p">()</span><span class="o">.</span><span class="n">minimize</span><span class="p">(</span><span class="n">cost</span><span class="p">)</span>

<span class="c1"># Accuracy</span>
<span class="n">correct_pred</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">equal</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span> <span class="n">tf</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span>
<span class="n">accuracy</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">reduce_mean</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">cast</span><span class="p">(</span><span class="n">correct_pred</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">),</span> <span class="n">name</span><span class="o">=</span><span class="s1">'accuracy'</span><span class="p">)</span>

<span class="n">tests</span><span class="o">.</span><span class="n">test_conv_net</span><span class="p">(</span><span class="n">conv_net</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>WARNING:tensorflow:From <ipython-input-12-8883713da052>:70: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See @{tf.nn.softmax_cross_entropy_with_logits_v2}.

Neural Network Built!
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

## Train the Neural Network[¶](#Train-the-Neural-Network)

### Single Optimization[¶](#Single-Optimization)

Implement the function `train_neural_network` to do a single optimization. The optimization should use `optimizer` to optimize in `session` with a `feed_dict` of the following:

*   `x` for image input
*   `y` for labels
*   `keep_prob` for keep probability for dropout

This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.

Note: Nothing needs to be returned. This function is only optimizing the neural network.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [13]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">train_neural_network</span><span class="p">(</span><span class="n">session</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">keep_probability</span><span class="p">,</span> <span class="n">feature_batch</span><span class="p">,</span> <span class="n">label_batch</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Optimize the session on a batch of images and labels</span>
 <span class="sd">: session: Current TensorFlow session</span>
 <span class="sd">: optimizer: TensorFlow optimizer function</span>
 <span class="sd">: keep_probability: keep probability</span>
 <span class="sd">: feature_batch: Batch of Numpy image data</span>
 <span class="sd">: label_batch: Batch of Numpy label data</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>
    <span class="n">session</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">optimizer</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">feature_batch</span><span class="p">,</span>
                                      <span class="n">y</span><span class="p">:</span> <span class="n">label_batch</span><span class="p">,</span>
                                      <span class="n">keep_prob</span><span class="p">:</span> <span class="n">keep_probability</span><span class="p">})</span>

<span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE</span>
<span class="sd">"""</span>
<span class="n">tests</span><span class="o">.</span><span class="n">test_train_nn</span><span class="p">(</span><span class="n">train_neural_network</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Tests Passed
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Show Stats[¶](#Show-Stats)

Implement the function `print_stats` to print loss and validation accuracy. Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy. Use a keep probability of `1.0` to calculate the loss and validation accuracy.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [14]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">print_stats</span><span class="p">(</span><span class="n">session</span><span class="p">,</span> <span class="n">feature_batch</span><span class="p">,</span> <span class="n">label_batch</span><span class="p">,</span> <span class="n">cost</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">):</span>
    <span class="sd">"""</span>
 <span class="sd">Print information about loss and validation accuracy</span>
 <span class="sd">: session: Current TensorFlow session</span>
 <span class="sd">: feature_batch: Batch of Numpy image data</span>
 <span class="sd">: label_batch: Batch of Numpy label data</span>
 <span class="sd">: cost: TensorFlow cost function</span>
 <span class="sd">: accuracy: TensorFlow accuracy function</span>
 <span class="sd">"""</span>
    <span class="c1"># TODO: Implement Function</span>
    <span class="n">loss</span><span class="o">=</span><span class="n">session</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">cost</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">feature_batch</span><span class="p">,</span>
                                      <span class="n">y</span><span class="p">:</span> <span class="n">label_batch</span><span class="p">,</span>
                                      <span class="n">keep_prob</span><span class="p">:</span> <span class="mi">1</span><span class="p">})</span>
    <span class="n">acc</span><span class="o">=</span><span class="n">session</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">accuracy</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">valid_features</span><span class="p">,</span>
                                      <span class="n">y</span><span class="p">:</span> <span class="n">valid_labels</span><span class="p">,</span>
                                      <span class="n">keep_prob</span><span class="p">:</span> <span class="mi">1</span><span class="p">})</span>

    <span class="nb">print</span><span class="p">(</span><span class="s1">'Cost:</span> <span class="si">{:.4f}</span><span class="s1">, accuracy:</span> <span class="si">{:.4f}</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">loss</span><span class="p">,</span> <span class="n">acc</span><span class="p">))</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Hyperparameters[¶](#Hyperparameters)

Tune the following parameters:

*   Set `epochs` to the number of iterations until the network stops learning or start overfitting
*   Set `batch_size` to the highest number that your machine has memory for. Most people set them to common sizes of memory:
    *   64
    *   128
    *   256
    *   ...
*   Set `keep_probability` to the probability of keeping a node using dropout

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [15]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1"># TODO: Tune Parameters</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="mi">75</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">256</span>
<span class="n">keep_probability</span> <span class="o">=</span> <span class="mf">0.75</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Train on a Single CIFAR-10 Batch[¶](#Train-on-a-Single-CIFAR-10-Batch)

Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy. Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [16]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL</span>
<span class="sd">"""</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'Checking the Training on a Single Batch...'</span><span class="p">)</span>
<span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">()</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
    <span class="c1"># Initializing the variables</span>
    <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">global_variables_initializer</span><span class="p">())</span>

    <span class="c1"># Training cycle</span>
    <span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">epochs</span><span class="p">):</span>
        <span class="n">batch_i</span> <span class="o">=</span> <span class="mi">1</span>
        <span class="k">for</span> <span class="n">batch_features</span><span class="p">,</span> <span class="n">batch_labels</span> <span class="ow">in</span> <span class="n">helper</span><span class="o">.</span><span class="n">load_preprocess_training_batch</span><span class="p">(</span><span class="n">batch_i</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
            <span class="n">train_neural_network</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">keep_probability</span><span class="p">,</span> <span class="n">batch_features</span><span class="p">,</span> <span class="n">batch_labels</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s1">'Epoch</span> <span class="si">{:>2}</span><span class="s1">, CIFAR-10 Batch</span> <span class="si">{}</span><span class="s1">:  '</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">batch_i</span><span class="p">),</span> <span class="n">end</span><span class="o">=</span><span class="s1">''</span><span class="p">)</span>
        <span class="n">print_stats</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">batch_features</span><span class="p">,</span> <span class="n">batch_labels</span><span class="p">,</span> <span class="n">cost</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Checking the Training on a Single Batch...
Epoch  1, CIFAR-10 Batch 1:  Cost: 2.1229, accuracy: 0.2314
Epoch  2, CIFAR-10 Batch 1:  Cost: 1.9428, accuracy: 0.3062
Epoch  3, CIFAR-10 Batch 1:  Cost: 1.7248, accuracy: 0.3742
Epoch  4, CIFAR-10 Batch 1:  Cost: 1.6253, accuracy: 0.3794
Epoch  5, CIFAR-10 Batch 1:  Cost: 1.2184, accuracy: 0.4392
Epoch  6, CIFAR-10 Batch 1:  Cost: 0.9963, accuracy: 0.4764
Epoch  7, CIFAR-10 Batch 1:  Cost: 0.8702, accuracy: 0.4728
Epoch  8, CIFAR-10 Batch 1:  Cost: 0.6748, accuracy: 0.4986
Epoch  9, CIFAR-10 Batch 1:  Cost: 0.5809, accuracy: 0.5024
Epoch 10, CIFAR-10 Batch 1:  Cost: 0.4781, accuracy: 0.5360
Epoch 11, CIFAR-10 Batch 1:  Cost: 0.3870, accuracy: 0.5516
Epoch 12, CIFAR-10 Batch 1:  Cost: 0.2857, accuracy: 0.5194
Epoch 13, CIFAR-10 Batch 1:  Cost: 0.2069, accuracy: 0.5286
Epoch 14, CIFAR-10 Batch 1:  Cost: 0.2167, accuracy: 0.5092
Epoch 15, CIFAR-10 Batch 1:  Cost: 0.1675, accuracy: 0.5428
Epoch 16, CIFAR-10 Batch 1:  Cost: 0.1396, accuracy: 0.5258
Epoch 17, CIFAR-10 Batch 1:  Cost: 0.1084, accuracy: 0.5500
Epoch 18, CIFAR-10 Batch 1:  Cost: 0.0642, accuracy: 0.5258
Epoch 19, CIFAR-10 Batch 1:  Cost: 0.0477, accuracy: 0.5308
Epoch 20, CIFAR-10 Batch 1:  Cost: 0.0676, accuracy: 0.4788
Epoch 21, CIFAR-10 Batch 1:  Cost: 0.0750, accuracy: 0.5116
Epoch 22, CIFAR-10 Batch 1:  Cost: 0.0402, accuracy: 0.5210
Epoch 23, CIFAR-10 Batch 1:  Cost: 0.0239, accuracy: 0.5236
Epoch 24, CIFAR-10 Batch 1:  Cost: 0.0238, accuracy: 0.5490
Epoch 25, CIFAR-10 Batch 1:  Cost: 0.0189, accuracy: 0.5468
Epoch 26, CIFAR-10 Batch 1:  Cost: 0.0218, accuracy: 0.5564
Epoch 27, CIFAR-10 Batch 1:  Cost: 0.0146, accuracy: 0.5458
Epoch 28, CIFAR-10 Batch 1:  Cost: 0.0083, accuracy: 0.5530
Epoch 29, CIFAR-10 Batch 1:  Cost: 0.0092, accuracy: 0.5664
Epoch 30, CIFAR-10 Batch 1:  Cost: 0.0032, accuracy: 0.5450
Epoch 31, CIFAR-10 Batch 1:  Cost: 0.0023, accuracy: 0.5066
Epoch 32, CIFAR-10 Batch 1:  Cost: 0.0213, accuracy: 0.5520
Epoch 33, CIFAR-10 Batch 1:  Cost: 0.0024, accuracy: 0.5604
Epoch 34, CIFAR-10 Batch 1:  Cost: 0.0004, accuracy: 0.5460
Epoch 35, CIFAR-10 Batch 1:  Cost: 0.0031, accuracy: 0.5386
Epoch 36, CIFAR-10 Batch 1:  Cost: 0.0017, accuracy: 0.5376
Epoch 37, CIFAR-10 Batch 1:  Cost: 0.0149, accuracy: 0.5222
Epoch 38, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.5454
Epoch 39, CIFAR-10 Batch 1:  Cost: 0.0008, accuracy: 0.5616
Epoch 40, CIFAR-10 Batch 1:  Cost: 0.0007, accuracy: 0.5548
Epoch 41, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.5386
Epoch 42, CIFAR-10 Batch 1:  Cost: 0.0013, accuracy: 0.5640
Epoch 43, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.5460
Epoch 44, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.5686
Epoch 45, CIFAR-10 Batch 1:  Cost: 0.0008, accuracy: 0.5438
Epoch 46, CIFAR-10 Batch 1:  Cost: 0.0006, accuracy: 0.5478
Epoch 47, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.5642
Epoch 48, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5588
Epoch 49, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.5466
Epoch 50, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.5750
Epoch 51, CIFAR-10 Batch 1:  Cost: 0.0448, accuracy: 0.5620
Epoch 52, CIFAR-10 Batch 1:  Cost: 0.0005, accuracy: 0.5494
Epoch 53, CIFAR-10 Batch 1:  Cost: 0.0006, accuracy: 0.5564
Epoch 54, CIFAR-10 Batch 1:  Cost: 0.0004, accuracy: 0.5644
Epoch 55, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5736
Epoch 56, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5666
Epoch 57, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.5686
Epoch 58, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.5622
Epoch 59, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.5810
Epoch 60, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5776
Epoch 61, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.5654
Epoch 62, CIFAR-10 Batch 1:  Cost: 0.0035, accuracy: 0.5722
Epoch 63, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5756
Epoch 64, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.5702
Epoch 65, CIFAR-10 Batch 1:  Cost: 0.0006, accuracy: 0.5728
Epoch 66, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5770
Epoch 67, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5812
Epoch 68, CIFAR-10 Batch 1:  Cost: 0.0011, accuracy: 0.5744
Epoch 69, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.5792
Epoch 70, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.5822
Epoch 71, CIFAR-10 Batch 1:  Cost: 0.0016, accuracy: 0.5692
Epoch 72, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.5742
Epoch 73, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.5842
Epoch 74, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.5808
Epoch 75, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.5784
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Fully Train the Model[¶](#Fully-Train-the-Model)

Now that you got a good accuracy with a single CIFAR-10 batch, try it with all five batches.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [17]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL</span>
<span class="sd">"""</span>
<span class="n">save_model_path</span> <span class="o">=</span> <span class="s1">'./image_classification'</span>

<span class="nb">print</span><span class="p">(</span><span class="s1">'Training...'</span><span class="p">)</span>
<span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">()</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
    <span class="c1"># Initializing the variables</span>
    <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">global_variables_initializer</span><span class="p">())</span>

    <span class="c1"># Training cycle</span>
    <span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">epochs</span><span class="p">):</span>
        <span class="c1"># Loop over all batches</span>
        <span class="n">n_batches</span> <span class="o">=</span> <span class="mi">5</span>
        <span class="k">for</span> <span class="n">batch_i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">n_batches</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
            <span class="k">for</span> <span class="n">batch_features</span><span class="p">,</span> <span class="n">batch_labels</span> <span class="ow">in</span> <span class="n">helper</span><span class="o">.</span><span class="n">load_preprocess_training_batch</span><span class="p">(</span><span class="n">batch_i</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
                <span class="n">train_neural_network</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">keep_probability</span><span class="p">,</span> <span class="n">batch_features</span><span class="p">,</span> <span class="n">batch_labels</span><span class="p">)</span>
            <span class="nb">print</span><span class="p">(</span><span class="s1">'Epoch</span> <span class="si">{:>2}</span><span class="s1">, CIFAR-10 Batch</span> <span class="si">{}</span><span class="s1">:  '</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">epoch</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">batch_i</span><span class="p">),</span> <span class="n">end</span><span class="o">=</span><span class="s1">''</span><span class="p">)</span>
            <span class="n">print_stats</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">batch_features</span><span class="p">,</span> <span class="n">batch_labels</span><span class="p">,</span> <span class="n">cost</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">)</span>

    <span class="c1"># Save Model</span>
    <span class="n">saver</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">Saver</span><span class="p">()</span>
    <span class="n">save_path</span> <span class="o">=</span> <span class="n">saver</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">save_model_path</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Training...
Epoch  1, CIFAR-10 Batch 1:  Cost: 2.2334, accuracy: 0.2128
Epoch  1, CIFAR-10 Batch 2:  Cost: 2.0022, accuracy: 0.3020
Epoch  1, CIFAR-10 Batch 3:  Cost: 1.7210, accuracy: 0.3518
Epoch  1, CIFAR-10 Batch 4:  Cost: 1.6453, accuracy: 0.4076
Epoch  1, CIFAR-10 Batch 5:  Cost: 1.5792, accuracy: 0.4210
Epoch  2, CIFAR-10 Batch 1:  Cost: 1.6177, accuracy: 0.4574
Epoch  2, CIFAR-10 Batch 2:  Cost: 1.3026, accuracy: 0.4664
Epoch  2, CIFAR-10 Batch 3:  Cost: 1.1262, accuracy: 0.5076
Epoch  2, CIFAR-10 Batch 4:  Cost: 1.2820, accuracy: 0.5208
Epoch  2, CIFAR-10 Batch 5:  Cost: 1.2221, accuracy: 0.5318
Epoch  3, CIFAR-10 Batch 1:  Cost: 1.2183, accuracy: 0.5446
Epoch  3, CIFAR-10 Batch 2:  Cost: 0.9502, accuracy: 0.5050
Epoch  3, CIFAR-10 Batch 3:  Cost: 0.8206, accuracy: 0.5188
Epoch  3, CIFAR-10 Batch 4:  Cost: 0.8890, accuracy: 0.5576
Epoch  3, CIFAR-10 Batch 5:  Cost: 0.9838, accuracy: 0.5248
Epoch  4, CIFAR-10 Batch 1:  Cost: 0.9229, accuracy: 0.5816
Epoch  4, CIFAR-10 Batch 2:  Cost: 0.7113, accuracy: 0.5644
Epoch  4, CIFAR-10 Batch 3:  Cost: 0.5572, accuracy: 0.5926
Epoch  4, CIFAR-10 Batch 4:  Cost: 0.6385, accuracy: 0.6032
Epoch  4, CIFAR-10 Batch 5:  Cost: 0.6238, accuracy: 0.5890
Epoch  5, CIFAR-10 Batch 1:  Cost: 0.7070, accuracy: 0.6172
Epoch  5, CIFAR-10 Batch 2:  Cost: 0.5191, accuracy: 0.6018
Epoch  5, CIFAR-10 Batch 3:  Cost: 0.4444, accuracy: 0.6120
Epoch  5, CIFAR-10 Batch 4:  Cost: 0.4955, accuracy: 0.6354
Epoch  5, CIFAR-10 Batch 5:  Cost: 0.4353, accuracy: 0.6380
Epoch  6, CIFAR-10 Batch 1:  Cost: 0.5556, accuracy: 0.6242
Epoch  6, CIFAR-10 Batch 2:  Cost: 0.3171, accuracy: 0.6416
Epoch  6, CIFAR-10 Batch 3:  Cost: 0.3173, accuracy: 0.6254
Epoch  6, CIFAR-10 Batch 4:  Cost: 0.4206, accuracy: 0.6406
Epoch  6, CIFAR-10 Batch 5:  Cost: 0.3020, accuracy: 0.6610
Epoch  7, CIFAR-10 Batch 1:  Cost: 0.4916, accuracy: 0.6092
Epoch  7, CIFAR-10 Batch 2:  Cost: 0.2334, accuracy: 0.6390
Epoch  7, CIFAR-10 Batch 3:  Cost: 0.2086, accuracy: 0.6556
Epoch  7, CIFAR-10 Batch 4:  Cost: 0.3295, accuracy: 0.6486
Epoch  7, CIFAR-10 Batch 5:  Cost: 0.2310, accuracy: 0.6602
Epoch  8, CIFAR-10 Batch 1:  Cost: 0.3007, accuracy: 0.6438
Epoch  8, CIFAR-10 Batch 2:  Cost: 0.2212, accuracy: 0.6644
Epoch  8, CIFAR-10 Batch 3:  Cost: 0.1533, accuracy: 0.6672
Epoch  8, CIFAR-10 Batch 4:  Cost: 0.1991, accuracy: 0.6532
Epoch  8, CIFAR-10 Batch 5:  Cost: 0.1632, accuracy: 0.6358
Epoch  9, CIFAR-10 Batch 1:  Cost: 0.2441, accuracy: 0.6476
Epoch  9, CIFAR-10 Batch 2:  Cost: 0.1850, accuracy: 0.6680
Epoch  9, CIFAR-10 Batch 3:  Cost: 0.1229, accuracy: 0.6486
Epoch  9, CIFAR-10 Batch 4:  Cost: 0.1235, accuracy: 0.6778
Epoch  9, CIFAR-10 Batch 5:  Cost: 0.1114, accuracy: 0.6746
Epoch 10, CIFAR-10 Batch 1:  Cost: 0.1968, accuracy: 0.6686
Epoch 10, CIFAR-10 Batch 2:  Cost: 0.1475, accuracy: 0.6614
Epoch 10, CIFAR-10 Batch 3:  Cost: 0.1188, accuracy: 0.6450
Epoch 10, CIFAR-10 Batch 4:  Cost: 0.1191, accuracy: 0.6618
Epoch 10, CIFAR-10 Batch 5:  Cost: 0.1125, accuracy: 0.6892
Epoch 11, CIFAR-10 Batch 1:  Cost: 0.1169, accuracy: 0.6208
Epoch 11, CIFAR-10 Batch 2:  Cost: 0.1224, accuracy: 0.6690
Epoch 11, CIFAR-10 Batch 3:  Cost: 0.0953, accuracy: 0.6592
Epoch 11, CIFAR-10 Batch 4:  Cost: 0.0951, accuracy: 0.6640
Epoch 11, CIFAR-10 Batch 5:  Cost: 0.0900, accuracy: 0.6752
Epoch 12, CIFAR-10 Batch 1:  Cost: 0.0674, accuracy: 0.6298
Epoch 12, CIFAR-10 Batch 2:  Cost: 0.0595, accuracy: 0.6692
Epoch 12, CIFAR-10 Batch 3:  Cost: 0.0516, accuracy: 0.6618
Epoch 12, CIFAR-10 Batch 4:  Cost: 0.0518, accuracy: 0.6666
Epoch 12, CIFAR-10 Batch 5:  Cost: 0.0537, accuracy: 0.6652
Epoch 13, CIFAR-10 Batch 1:  Cost: 0.0788, accuracy: 0.6478
Epoch 13, CIFAR-10 Batch 2:  Cost: 0.0398, accuracy: 0.6632
Epoch 13, CIFAR-10 Batch 3:  Cost: 0.0316, accuracy: 0.6646
Epoch 13, CIFAR-10 Batch 4:  Cost: 0.0251, accuracy: 0.6828
Epoch 13, CIFAR-10 Batch 5:  Cost: 0.0327, accuracy: 0.6750
Epoch 14, CIFAR-10 Batch 1:  Cost: 0.0403, accuracy: 0.6772
Epoch 14, CIFAR-10 Batch 2:  Cost: 0.0394, accuracy: 0.6552
Epoch 14, CIFAR-10 Batch 3:  Cost: 0.0107, accuracy: 0.6680
Epoch 14, CIFAR-10 Batch 4:  Cost: 0.0259, accuracy: 0.6796
Epoch 14, CIFAR-10 Batch 5:  Cost: 0.0361, accuracy: 0.6568
Epoch 15, CIFAR-10 Batch 1:  Cost: 0.0240, accuracy: 0.6700
Epoch 15, CIFAR-10 Batch 2:  Cost: 0.0290, accuracy: 0.6508
Epoch 15, CIFAR-10 Batch 3:  Cost: 0.0094, accuracy: 0.6596
Epoch 15, CIFAR-10 Batch 4:  Cost: 0.0371, accuracy: 0.6696
Epoch 15, CIFAR-10 Batch 5:  Cost: 0.0195, accuracy: 0.6920
Epoch 16, CIFAR-10 Batch 1:  Cost: 0.0131, accuracy: 0.6760
Epoch 16, CIFAR-10 Batch 2:  Cost: 0.0154, accuracy: 0.6362
Epoch 16, CIFAR-10 Batch 3:  Cost: 0.0169, accuracy: 0.6580
Epoch 16, CIFAR-10 Batch 4:  Cost: 0.0207, accuracy: 0.6740
Epoch 16, CIFAR-10 Batch 5:  Cost: 0.0165, accuracy: 0.6688
Epoch 17, CIFAR-10 Batch 1:  Cost: 0.0166, accuracy: 0.6688
Epoch 17, CIFAR-10 Batch 2:  Cost: 0.0309, accuracy: 0.6200
Epoch 17, CIFAR-10 Batch 3:  Cost: 0.0107, accuracy: 0.6822
Epoch 17, CIFAR-10 Batch 4:  Cost: 0.0200, accuracy: 0.6858
Epoch 17, CIFAR-10 Batch 5:  Cost: 0.0058, accuracy: 0.6744
Epoch 18, CIFAR-10 Batch 1:  Cost: 0.0085, accuracy: 0.6762
Epoch 18, CIFAR-10 Batch 2:  Cost: 0.0149, accuracy: 0.6450
Epoch 18, CIFAR-10 Batch 3:  Cost: 0.0134, accuracy: 0.6818
Epoch 18, CIFAR-10 Batch 4:  Cost: 0.0110, accuracy: 0.6896
Epoch 18, CIFAR-10 Batch 5:  Cost: 0.0136, accuracy: 0.6600
Epoch 19, CIFAR-10 Batch 1:  Cost: 0.0261, accuracy: 0.6682
Epoch 19, CIFAR-10 Batch 2:  Cost: 0.0121, accuracy: 0.6768
Epoch 19, CIFAR-10 Batch 3:  Cost: 0.0069, accuracy: 0.6756
Epoch 19, CIFAR-10 Batch 4:  Cost: 0.0052, accuracy: 0.6756
Epoch 19, CIFAR-10 Batch 5:  Cost: 0.0044, accuracy: 0.6702
Epoch 20, CIFAR-10 Batch 1:  Cost: 0.0059, accuracy: 0.6796
Epoch 20, CIFAR-10 Batch 2:  Cost: 0.0062, accuracy: 0.6640
Epoch 20, CIFAR-10 Batch 3:  Cost: 0.0039, accuracy: 0.6892
Epoch 20, CIFAR-10 Batch 4:  Cost: 0.0083, accuracy: 0.6834
Epoch 20, CIFAR-10 Batch 5:  Cost: 0.0033, accuracy: 0.6730
Epoch 21, CIFAR-10 Batch 1:  Cost: 0.0106, accuracy: 0.6684
Epoch 21, CIFAR-10 Batch 2:  Cost: 0.0096, accuracy: 0.6870
Epoch 21, CIFAR-10 Batch 3:  Cost: 0.0075, accuracy: 0.6818
Epoch 21, CIFAR-10 Batch 4:  Cost: 0.0009, accuracy: 0.6832
Epoch 21, CIFAR-10 Batch 5:  Cost: 0.0043, accuracy: 0.6844
Epoch 22, CIFAR-10 Batch 1:  Cost: 0.0080, accuracy: 0.6584
Epoch 22, CIFAR-10 Batch 2:  Cost: 0.0078, accuracy: 0.6766
Epoch 22, CIFAR-10 Batch 3:  Cost: 0.0013, accuracy: 0.6904
Epoch 22, CIFAR-10 Batch 4:  Cost: 0.0036, accuracy: 0.6828
Epoch 22, CIFAR-10 Batch 5:  Cost: 0.0068, accuracy: 0.6718
Epoch 23, CIFAR-10 Batch 1:  Cost: 0.0065, accuracy: 0.6760
Epoch 23, CIFAR-10 Batch 2:  Cost: 0.0039, accuracy: 0.6656
Epoch 23, CIFAR-10 Batch 3:  Cost: 0.0007, accuracy: 0.6824
Epoch 23, CIFAR-10 Batch 4:  Cost: 0.0063, accuracy: 0.6700
Epoch 23, CIFAR-10 Batch 5:  Cost: 0.0061, accuracy: 0.6786
Epoch 24, CIFAR-10 Batch 1:  Cost: 0.0046, accuracy: 0.6762
Epoch 24, CIFAR-10 Batch 2:  Cost: 0.0058, accuracy: 0.6690
Epoch 24, CIFAR-10 Batch 3:  Cost: 0.0031, accuracy: 0.6778
Epoch 24, CIFAR-10 Batch 4:  Cost: 0.0086, accuracy: 0.6844
Epoch 24, CIFAR-10 Batch 5:  Cost: 0.0042, accuracy: 0.6730
Epoch 25, CIFAR-10 Batch 1:  Cost: 0.0017, accuracy: 0.6682
Epoch 25, CIFAR-10 Batch 2:  Cost: 0.0019, accuracy: 0.6534
Epoch 25, CIFAR-10 Batch 3:  Cost: 0.0015, accuracy: 0.6556
Epoch 25, CIFAR-10 Batch 4:  Cost: 0.0043, accuracy: 0.6816
Epoch 25, CIFAR-10 Batch 5:  Cost: 0.0063, accuracy: 0.6814
Epoch 26, CIFAR-10 Batch 1:  Cost: 0.0068, accuracy: 0.6606
Epoch 26, CIFAR-10 Batch 2:  Cost: 0.0039, accuracy: 0.6668
Epoch 26, CIFAR-10 Batch 3:  Cost: 0.0022, accuracy: 0.6756
Epoch 26, CIFAR-10 Batch 4:  Cost: 0.0109, accuracy: 0.6674
Epoch 26, CIFAR-10 Batch 5:  Cost: 0.0018, accuracy: 0.6750
Epoch 27, CIFAR-10 Batch 1:  Cost: 0.0049, accuracy: 0.6606
Epoch 27, CIFAR-10 Batch 2:  Cost: 0.0099, accuracy: 0.6720
Epoch 27, CIFAR-10 Batch 3:  Cost: 0.0004, accuracy: 0.6880
Epoch 27, CIFAR-10 Batch 4:  Cost: 0.0018, accuracy: 0.6822
Epoch 27, CIFAR-10 Batch 5:  Cost: 0.0069, accuracy: 0.6668
Epoch 28, CIFAR-10 Batch 1:  Cost: 0.0005, accuracy: 0.6748
Epoch 28, CIFAR-10 Batch 2:  Cost: 0.0003, accuracy: 0.6666
Epoch 28, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.6834
Epoch 28, CIFAR-10 Batch 4:  Cost: 0.0004, accuracy: 0.6712
Epoch 28, CIFAR-10 Batch 5:  Cost: 0.0197, accuracy: 0.6816
Epoch 29, CIFAR-10 Batch 1:  Cost: 0.0166, accuracy: 0.6774
Epoch 29, CIFAR-10 Batch 2:  Cost: 0.0050, accuracy: 0.6756
Epoch 29, CIFAR-10 Batch 3:  Cost: 0.0004, accuracy: 0.6742
Epoch 29, CIFAR-10 Batch 4:  Cost: 0.0004, accuracy: 0.6862
Epoch 29, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.6954
Epoch 30, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.6902
Epoch 30, CIFAR-10 Batch 2:  Cost: 0.0008, accuracy: 0.6740
Epoch 30, CIFAR-10 Batch 3:  Cost: 0.0004, accuracy: 0.6848
Epoch 30, CIFAR-10 Batch 4:  Cost: 0.0011, accuracy: 0.6902
Epoch 30, CIFAR-10 Batch 5:  Cost: 0.0012, accuracy: 0.6794
Epoch 31, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6746
Epoch 31, CIFAR-10 Batch 2:  Cost: 0.0058, accuracy: 0.6816
Epoch 31, CIFAR-10 Batch 3:  Cost: 0.0002, accuracy: 0.6672
Epoch 31, CIFAR-10 Batch 4:  Cost: 0.0020, accuracy: 0.6894
Epoch 31, CIFAR-10 Batch 5:  Cost: 0.0017, accuracy: 0.6864
Epoch 32, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6834
Epoch 32, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.6856
Epoch 32, CIFAR-10 Batch 3:  Cost: 0.0004, accuracy: 0.6832
Epoch 32, CIFAR-10 Batch 4:  Cost: 0.0008, accuracy: 0.6726
Epoch 32, CIFAR-10 Batch 5:  Cost: 0.0006, accuracy: 0.6926
Epoch 33, CIFAR-10 Batch 1:  Cost: 0.0008, accuracy: 0.6912
Epoch 33, CIFAR-10 Batch 2:  Cost: 0.0006, accuracy: 0.6654
Epoch 33, CIFAR-10 Batch 3:  Cost: 0.0022, accuracy: 0.6866
Epoch 33, CIFAR-10 Batch 4:  Cost: 0.0011, accuracy: 0.6860
Epoch 33, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.6938
Epoch 34, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.6930
Epoch 34, CIFAR-10 Batch 2:  Cost: 0.0006, accuracy: 0.7018
Epoch 34, CIFAR-10 Batch 3:  Cost: 0.0035, accuracy: 0.6802
Epoch 34, CIFAR-10 Batch 4:  Cost: 0.0023, accuracy: 0.6822
Epoch 34, CIFAR-10 Batch 5:  Cost: 0.0007, accuracy: 0.6856
Epoch 35, CIFAR-10 Batch 1:  Cost: 0.0005, accuracy: 0.6912
Epoch 35, CIFAR-10 Batch 2:  Cost: 0.0004, accuracy: 0.6864
Epoch 35, CIFAR-10 Batch 3:  Cost: 0.0004, accuracy: 0.6830
Epoch 35, CIFAR-10 Batch 4:  Cost: 0.0036, accuracy: 0.6838
Epoch 35, CIFAR-10 Batch 5:  Cost: 0.0013, accuracy: 0.6874
Epoch 36, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6706
Epoch 36, CIFAR-10 Batch 2:  Cost: 0.0002, accuracy: 0.6980
Epoch 36, CIFAR-10 Batch 3:  Cost: 0.0060, accuracy: 0.6780
Epoch 36, CIFAR-10 Batch 4:  Cost: 0.0007, accuracy: 0.6734
Epoch 36, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6922
Epoch 37, CIFAR-10 Batch 1:  Cost: 0.0004, accuracy: 0.6726
Epoch 37, CIFAR-10 Batch 2:  Cost: 0.0000, accuracy: 0.6966
Epoch 37, CIFAR-10 Batch 3:  Cost: 0.0003, accuracy: 0.6914
Epoch 37, CIFAR-10 Batch 4:  Cost: 0.0005, accuracy: 0.6936
Epoch 37, CIFAR-10 Batch 5:  Cost: 0.0000, accuracy: 0.6892
Epoch 38, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6842
Epoch 38, CIFAR-10 Batch 2:  Cost: 0.0000, accuracy: 0.6912
Epoch 38, CIFAR-10 Batch 3:  Cost: 0.0022, accuracy: 0.6848
Epoch 38, CIFAR-10 Batch 4:  Cost: 0.0001, accuracy: 0.6940
Epoch 38, CIFAR-10 Batch 5:  Cost: 0.0008, accuracy: 0.6934
Epoch 39, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.6806
Epoch 39, CIFAR-10 Batch 2:  Cost: 0.0002, accuracy: 0.6862
Epoch 39, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6848
Epoch 39, CIFAR-10 Batch 4:  Cost: 0.0005, accuracy: 0.6866
Epoch 39, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.6844
Epoch 40, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.6830
Epoch 40, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.6958
Epoch 40, CIFAR-10 Batch 3:  Cost: 0.0003, accuracy: 0.6806
Epoch 40, CIFAR-10 Batch 4:  Cost: 0.0002, accuracy: 0.6880
Epoch 40, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.6928
Epoch 41, CIFAR-10 Batch 1:  Cost: 0.0010, accuracy: 0.6696
Epoch 41, CIFAR-10 Batch 2:  Cost: 0.0002, accuracy: 0.6846
Epoch 41, CIFAR-10 Batch 3:  Cost: 0.0025, accuracy: 0.6842
Epoch 41, CIFAR-10 Batch 4:  Cost: 0.0038, accuracy: 0.6982
Epoch 41, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6982
Epoch 42, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.6816
Epoch 42, CIFAR-10 Batch 2:  Cost: 0.0024, accuracy: 0.6976
Epoch 42, CIFAR-10 Batch 3:  Cost: 0.0006, accuracy: 0.6900
Epoch 42, CIFAR-10 Batch 4:  Cost: 0.0017, accuracy: 0.6922
Epoch 42, CIFAR-10 Batch 5:  Cost: 0.0008, accuracy: 0.6948
Epoch 43, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.7012
Epoch 43, CIFAR-10 Batch 2:  Cost: 0.0041, accuracy: 0.6982
Epoch 43, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.6926
Epoch 43, CIFAR-10 Batch 4:  Cost: 0.0005, accuracy: 0.6724
Epoch 43, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.6898
Epoch 44, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.6860
Epoch 44, CIFAR-10 Batch 2:  Cost: 0.0004, accuracy: 0.6982
Epoch 44, CIFAR-10 Batch 3:  Cost: 0.0002, accuracy: 0.6772
Epoch 44, CIFAR-10 Batch 4:  Cost: 0.0004, accuracy: 0.6998
Epoch 44, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6984
Epoch 45, CIFAR-10 Batch 1:  Cost: 0.0008, accuracy: 0.6896
Epoch 45, CIFAR-10 Batch 2:  Cost: 0.0000, accuracy: 0.6980
Epoch 45, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6840
Epoch 45, CIFAR-10 Batch 4:  Cost: 0.0006, accuracy: 0.6974
Epoch 45, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6950
Epoch 46, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6980
Epoch 46, CIFAR-10 Batch 2:  Cost: 0.0012, accuracy: 0.6954
Epoch 46, CIFAR-10 Batch 3:  Cost: 0.0022, accuracy: 0.6894
Epoch 46, CIFAR-10 Batch 4:  Cost: 0.0017, accuracy: 0.6888
Epoch 46, CIFAR-10 Batch 5:  Cost: 0.0000, accuracy: 0.7004
Epoch 47, CIFAR-10 Batch 1:  Cost: 0.0011, accuracy: 0.6992
Epoch 47, CIFAR-10 Batch 2:  Cost: 0.0003, accuracy: 0.6962
Epoch 47, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.6898
Epoch 47, CIFAR-10 Batch 4:  Cost: 0.0018, accuracy: 0.6852
Epoch 47, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6786
Epoch 48, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6996
Epoch 48, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.6946
Epoch 48, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6802
Epoch 48, CIFAR-10 Batch 4:  Cost: 0.0025, accuracy: 0.6962
Epoch 48, CIFAR-10 Batch 5:  Cost: 0.0003, accuracy: 0.7062
Epoch 49, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.7052
Epoch 49, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.6894
Epoch 49, CIFAR-10 Batch 3:  Cost: 0.0014, accuracy: 0.6886
Epoch 49, CIFAR-10 Batch 4:  Cost: 0.0004, accuracy: 0.6938
Epoch 49, CIFAR-10 Batch 5:  Cost: 0.0002, accuracy: 0.6898
Epoch 50, CIFAR-10 Batch 1:  Cost: 0.0027, accuracy: 0.7026
Epoch 50, CIFAR-10 Batch 2:  Cost: 0.0000, accuracy: 0.7020
Epoch 50, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.6924
Epoch 50, CIFAR-10 Batch 4:  Cost: 0.0000, accuracy: 0.6954
Epoch 50, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.7048
Epoch 51, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.7110
Epoch 51, CIFAR-10 Batch 2:  Cost: 0.0009, accuracy: 0.6926
Epoch 51, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6970
Epoch 51, CIFAR-10 Batch 4:  Cost: 0.0003, accuracy: 0.6942
Epoch 51, CIFAR-10 Batch 5:  Cost: 0.0009, accuracy: 0.6916
Epoch 52, CIFAR-10 Batch 1:  Cost: 0.0007, accuracy: 0.7016
Epoch 52, CIFAR-10 Batch 2:  Cost: 0.0004, accuracy: 0.7056
Epoch 52, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6978
Epoch 52, CIFAR-10 Batch 4:  Cost: 0.0002, accuracy: 0.7040
Epoch 52, CIFAR-10 Batch 5:  Cost: 0.0051, accuracy: 0.7032
Epoch 53, CIFAR-10 Batch 1:  Cost: 0.0017, accuracy: 0.7012
Epoch 53, CIFAR-10 Batch 2:  Cost: 0.0003, accuracy: 0.6862
Epoch 53, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6940
Epoch 53, CIFAR-10 Batch 4:  Cost: 0.0016, accuracy: 0.6988
Epoch 53, CIFAR-10 Batch 5:  Cost: 0.0008, accuracy: 0.6950
Epoch 54, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6970
Epoch 54, CIFAR-10 Batch 2:  Cost: 0.0007, accuracy: 0.6934
Epoch 54, CIFAR-10 Batch 3:  Cost: 0.0004, accuracy: 0.6962
Epoch 54, CIFAR-10 Batch 4:  Cost: 0.0003, accuracy: 0.6984
Epoch 54, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6962
Epoch 55, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.7084
Epoch 55, CIFAR-10 Batch 2:  Cost: 0.0020, accuracy: 0.6922
Epoch 55, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6824
Epoch 55, CIFAR-10 Batch 4:  Cost: 0.0020, accuracy: 0.6918
Epoch 55, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6934
Epoch 56, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.7092
Epoch 56, CIFAR-10 Batch 2:  Cost: 0.0002, accuracy: 0.6974
Epoch 56, CIFAR-10 Batch 3:  Cost: 0.0006, accuracy: 0.6688
Epoch 56, CIFAR-10 Batch 4:  Cost: 0.0001, accuracy: 0.6918
Epoch 56, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6954
Epoch 57, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.7142
Epoch 57, CIFAR-10 Batch 2:  Cost: 0.0003, accuracy: 0.7058
Epoch 57, CIFAR-10 Batch 3:  Cost: 0.0002, accuracy: 0.6942
Epoch 57, CIFAR-10 Batch 4:  Cost: 0.0002, accuracy: 0.6982
Epoch 57, CIFAR-10 Batch 5:  Cost: 0.0008, accuracy: 0.6964
Epoch 58, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.7088
Epoch 58, CIFAR-10 Batch 2:  Cost: 0.0004, accuracy: 0.7072
Epoch 58, CIFAR-10 Batch 3:  Cost: 0.0002, accuracy: 0.6926
Epoch 58, CIFAR-10 Batch 4:  Cost: 0.0002, accuracy: 0.6864
Epoch 58, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.6854
Epoch 59, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.7014
Epoch 59, CIFAR-10 Batch 2:  Cost: 0.0004, accuracy: 0.7010
Epoch 59, CIFAR-10 Batch 3:  Cost: 0.0002, accuracy: 0.6844
Epoch 59, CIFAR-10 Batch 4:  Cost: 0.0005, accuracy: 0.6932
Epoch 59, CIFAR-10 Batch 5:  Cost: 0.0007, accuracy: 0.7004
Epoch 60, CIFAR-10 Batch 1:  Cost: 0.0006, accuracy: 0.7056
Epoch 60, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.6954
Epoch 60, CIFAR-10 Batch 3:  Cost: 0.0076, accuracy: 0.6896
Epoch 60, CIFAR-10 Batch 4:  Cost: 0.0012, accuracy: 0.6942
Epoch 60, CIFAR-10 Batch 5:  Cost: 0.0000, accuracy: 0.7078
Epoch 61, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6988
Epoch 61, CIFAR-10 Batch 2:  Cost: 0.0007, accuracy: 0.6998
Epoch 61, CIFAR-10 Batch 3:  Cost: 0.0018, accuracy: 0.6794
Epoch 61, CIFAR-10 Batch 4:  Cost: 0.0011, accuracy: 0.6956
Epoch 61, CIFAR-10 Batch 5:  Cost: 0.0000, accuracy: 0.7072
Epoch 62, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.7088
Epoch 62, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.7014
Epoch 62, CIFAR-10 Batch 3:  Cost: 0.0002, accuracy: 0.6924
Epoch 62, CIFAR-10 Batch 4:  Cost: 0.0001, accuracy: 0.6902
Epoch 62, CIFAR-10 Batch 5:  Cost: 0.0002, accuracy: 0.6934
Epoch 63, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.7120
Epoch 63, CIFAR-10 Batch 2:  Cost: 0.0000, accuracy: 0.7000
Epoch 63, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6912
Epoch 63, CIFAR-10 Batch 4:  Cost: 0.0003, accuracy: 0.6878
Epoch 63, CIFAR-10 Batch 5:  Cost: 0.0088, accuracy: 0.6832
Epoch 64, CIFAR-10 Batch 1:  Cost: 0.0007, accuracy: 0.7116
Epoch 64, CIFAR-10 Batch 2:  Cost: 0.0000, accuracy: 0.7136
Epoch 64, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6892
Epoch 64, CIFAR-10 Batch 4:  Cost: 0.0007, accuracy: 0.6920
Epoch 64, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.7062
Epoch 65, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.7050
Epoch 65, CIFAR-10 Batch 2:  Cost: 0.0003, accuracy: 0.6984
Epoch 65, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.7024
Epoch 65, CIFAR-10 Batch 4:  Cost: 0.0000, accuracy: 0.6856
Epoch 65, CIFAR-10 Batch 5:  Cost: 0.0005, accuracy: 0.7008
Epoch 66, CIFAR-10 Batch 1:  Cost: 0.0007, accuracy: 0.7116
Epoch 66, CIFAR-10 Batch 2:  Cost: 0.0013, accuracy: 0.7088
Epoch 66, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.6918
Epoch 66, CIFAR-10 Batch 4:  Cost: 0.0000, accuracy: 0.6912
Epoch 66, CIFAR-10 Batch 5:  Cost: 0.0002, accuracy: 0.6936
Epoch 67, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.6988
Epoch 67, CIFAR-10 Batch 2:  Cost: 0.0000, accuracy: 0.7070
Epoch 67, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6918
Epoch 67, CIFAR-10 Batch 4:  Cost: 0.0001, accuracy: 0.6922
Epoch 67, CIFAR-10 Batch 5:  Cost: 0.0009, accuracy: 0.7050
Epoch 68, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.7048
Epoch 68, CIFAR-10 Batch 2:  Cost: 0.0055, accuracy: 0.7042
Epoch 68, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.7104
Epoch 68, CIFAR-10 Batch 4:  Cost: 0.0003, accuracy: 0.6804
Epoch 68, CIFAR-10 Batch 5:  Cost: 0.0003, accuracy: 0.7044
Epoch 69, CIFAR-10 Batch 1:  Cost: 0.0003, accuracy: 0.7074
Epoch 69, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.7076
Epoch 69, CIFAR-10 Batch 3:  Cost: 0.0025, accuracy: 0.6894
Epoch 69, CIFAR-10 Batch 4:  Cost: 0.0000, accuracy: 0.6906
Epoch 69, CIFAR-10 Batch 5:  Cost: 0.0000, accuracy: 0.7158
Epoch 70, CIFAR-10 Batch 1:  Cost: 0.0002, accuracy: 0.6974
Epoch 70, CIFAR-10 Batch 2:  Cost: 0.0013, accuracy: 0.7028
Epoch 70, CIFAR-10 Batch 3:  Cost: 0.0000, accuracy: 0.6946
Epoch 70, CIFAR-10 Batch 4:  Cost: 0.0002, accuracy: 0.6856
Epoch 70, CIFAR-10 Batch 5:  Cost: 0.0004, accuracy: 0.7082
Epoch 71, CIFAR-10 Batch 1:  Cost: 0.0001, accuracy: 0.7028
Epoch 71, CIFAR-10 Batch 2:  Cost: 0.0001, accuracy: 0.7020
Epoch 71, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.7016
Epoch 71, CIFAR-10 Batch 4:  Cost: 0.0208, accuracy: 0.6844
Epoch 71, CIFAR-10 Batch 5:  Cost: 0.0003, accuracy: 0.7014
Epoch 72, CIFAR-10 Batch 1:  Cost: 0.0006, accuracy: 0.7110
Epoch 72, CIFAR-10 Batch 2:  Cost: 0.0014, accuracy: 0.7076
Epoch 72, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.6852
Epoch 72, CIFAR-10 Batch 4:  Cost: 0.0005, accuracy: 0.6868
Epoch 72, CIFAR-10 Batch 5:  Cost: 0.0002, accuracy: 0.7000
Epoch 73, CIFAR-10 Batch 1:  Cost: 0.0000, accuracy: 0.7094
Epoch 73, CIFAR-10 Batch 2:  Cost: 0.0907, accuracy: 0.7058
Epoch 73, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.7026
Epoch 73, CIFAR-10 Batch 4:  Cost: 0.0042, accuracy: 0.6886
Epoch 73, CIFAR-10 Batch 5:  Cost: 0.0000, accuracy: 0.7108
Epoch 74, CIFAR-10 Batch 1:  Cost: 0.0019, accuracy: 0.7042
Epoch 74, CIFAR-10 Batch 2:  Cost: 0.0010, accuracy: 0.7050
Epoch 74, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.7138
Epoch 74, CIFAR-10 Batch 4:  Cost: 0.0006, accuracy: 0.6692
Epoch 74, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.6912
Epoch 75, CIFAR-10 Batch 1:  Cost: 0.0104, accuracy: 0.7080
Epoch 75, CIFAR-10 Batch 2:  Cost: 0.0004, accuracy: 0.7064
Epoch 75, CIFAR-10 Batch 3:  Cost: 0.0001, accuracy: 0.7064
Epoch 75, CIFAR-10 Batch 4:  Cost: 0.0002, accuracy: 0.6866
Epoch 75, CIFAR-10 Batch 5:  Cost: 0.0001, accuracy: 0.7094
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

# Checkpoint[¶](#Checkpoint)

The model has been saved to disk.

## Test Model[¶](#Test-Model)

Test your model against the test dataset. This will be your final accuracy. You should have an accuracy greater than 50%. If you don't, keep tweaking the model architecture and parameters.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [18]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="sd">"""</span>
<span class="sd">DON'T MODIFY ANYTHING IN THIS CELL</span>
<span class="sd">"""</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="o">%</span><span class="k">config</span> InlineBackend.figure_format = 'retina'

<span class="kn">import</span> <span class="nn">tensorflow</span> <span class="k">as</span> <span class="nn">tf</span>
<span class="kn">import</span> <span class="nn">pickle</span>
<span class="kn">import</span> <span class="nn">helper</span>
<span class="kn">import</span> <span class="nn">random</span>

<span class="c1"># Set batch size if not already set</span>
<span class="k">try</span><span class="p">:</span>
    <span class="k">if</span> <span class="n">batch_size</span><span class="p">:</span>
        <span class="k">pass</span>
<span class="k">except</span> <span class="ne">NameError</span><span class="p">:</span>
    <span class="n">batch_size</span> <span class="o">=</span> <span class="mi">64</span>

<span class="n">save_model_path</span> <span class="o">=</span> <span class="s1">'./image_classification'</span>
<span class="n">n_samples</span> <span class="o">=</span> <span class="mi">4</span>
<span class="n">top_n_predictions</span> <span class="o">=</span> <span class="mi">3</span>

<span class="k">def</span> <span class="nf">test_model</span><span class="p">():</span>
    <span class="sd">"""</span>
 <span class="sd">Test the saved model against the test dataset</span>
 <span class="sd">"""</span>

    <span class="n">test_features</span><span class="p">,</span> <span class="n">test_labels</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="nb">open</span><span class="p">(</span><span class="s1">'preprocess_training.p'</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">'rb'</span><span class="p">))</span>
    <span class="n">loaded_graph</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Graph</span><span class="p">()</span>

    <span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">(</span><span class="n">graph</span><span class="o">=</span><span class="n">loaded_graph</span><span class="p">)</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
        <span class="c1"># Load model</span>
        <span class="n">loader</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">import_meta_graph</span><span class="p">(</span><span class="n">save_model_path</span> <span class="o">+</span> <span class="s1">'.meta'</span><span class="p">)</span>
        <span class="n">loader</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">save_model_path</span><span class="p">)</span>

        <span class="c1"># Get Tensors from loaded model</span>
        <span class="n">loaded_x</span> <span class="o">=</span> <span class="n">loaded_graph</span><span class="o">.</span><span class="n">get_tensor_by_name</span><span class="p">(</span><span class="s1">'x:0'</span><span class="p">)</span>
        <span class="n">loaded_y</span> <span class="o">=</span> <span class="n">loaded_graph</span><span class="o">.</span><span class="n">get_tensor_by_name</span><span class="p">(</span><span class="s1">'y:0'</span><span class="p">)</span>
        <span class="n">loaded_keep_prob</span> <span class="o">=</span> <span class="n">loaded_graph</span><span class="o">.</span><span class="n">get_tensor_by_name</span><span class="p">(</span><span class="s1">'keep_prob:0'</span><span class="p">)</span>
        <span class="n">loaded_logits</span> <span class="o">=</span> <span class="n">loaded_graph</span><span class="o">.</span><span class="n">get_tensor_by_name</span><span class="p">(</span><span class="s1">'logits:0'</span><span class="p">)</span>
        <span class="n">loaded_acc</span> <span class="o">=</span> <span class="n">loaded_graph</span><span class="o">.</span><span class="n">get_tensor_by_name</span><span class="p">(</span><span class="s1">'accuracy:0'</span><span class="p">)</span>

        <span class="c1"># Get accuracy in batches for memory limitations</span>
        <span class="n">test_batch_acc_total</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="n">test_batch_count</span> <span class="o">=</span> <span class="mi">0</span>

        <span class="k">for</span> <span class="n">train_feature_batch</span><span class="p">,</span> <span class="n">train_label_batch</span> <span class="ow">in</span> <span class="n">helper</span><span class="o">.</span><span class="n">batch_features_labels</span><span class="p">(</span><span class="n">test_features</span><span class="p">,</span> <span class="n">test_labels</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
            <span class="n">test_batch_acc_total</span> <span class="o">+=</span> <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
                <span class="n">loaded_acc</span><span class="p">,</span>
                <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">loaded_x</span><span class="p">:</span> <span class="n">train_feature_batch</span><span class="p">,</span> <span class="n">loaded_y</span><span class="p">:</span> <span class="n">train_label_batch</span><span class="p">,</span> <span class="n">loaded_keep_prob</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">})</span>
            <span class="n">test_batch_count</span> <span class="o">+=</span> <span class="mi">1</span>

        <span class="nb">print</span><span class="p">(</span><span class="s1">'Testing Accuracy:</span> <span class="si">{}</span><span class="se">\n</span><span class="s1">'</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">test_batch_acc_total</span><span class="o">/</span><span class="n">test_batch_count</span><span class="p">))</span>

        <span class="c1"># Print Random Samples</span>
        <span class="n">random_test_features</span><span class="p">,</span> <span class="n">random_test_labels</span> <span class="o">=</span> <span class="nb">tuple</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="o">*</span><span class="n">random</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="nb">list</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">test_features</span><span class="p">,</span> <span class="n">test_labels</span><span class="p">)),</span> <span class="n">n_samples</span><span class="p">)))</span>
        <span class="n">random_test_predictions</span> <span class="o">=</span> <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
            <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">top_k</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">loaded_logits</span><span class="p">),</span> <span class="n">top_n_predictions</span><span class="p">),</span>
            <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">loaded_x</span><span class="p">:</span> <span class="n">random_test_features</span><span class="p">,</span> <span class="n">loaded_y</span><span class="p">:</span> <span class="n">random_test_labels</span><span class="p">,</span> <span class="n">loaded_keep_prob</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">})</span>
        <span class="n">helper</span><span class="o">.</span><span class="n">display_image_predictions</span><span class="p">(</span><span class="n">random_test_features</span><span class="p">,</span> <span class="n">random_test_labels</span><span class="p">,</span> <span class="n">random_test_predictions</span><span class="p">)</span>

<span class="n">test_model</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>INFO:tensorflow:Restoring parameters from ./image_classification
Testing Accuracy: 0.70361328125

</pre>

</div>

</div>

<div class="output_area">


</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

## Why 50-80% Accuracy?[¶](#Why-50-80%-Accuracy?)

You might be wondering why you can't get an accuracy any higher. First things first, 50% isn't bad for a simple CNN. Pure guessing would get you 10% accuracy. That's because there are many more techniques that can be applied to your model and we recemmond that once you are done with this project, you explore!

## Submitting This Project[¶](#Submitting-This-Project)

When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "image_classification.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.

</div>

</div>

</div>

</div>

</div>
