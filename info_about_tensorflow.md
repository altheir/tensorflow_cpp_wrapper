###Purpose
Explain a few fundamentals about how the c++ api works.

#### Tensor
Fundamental object to tensorflow. It is essentialy an N-dimensional array.

#### Operations
Operations perform certain tasks on a tensor object. For example, a div tensorflow::op
will divide every object in the tensor.


#### Graphs
Most everything in tensorflow is expressed through a graph format.
A graph is a collection of operations that will operate on tensors, and will often return a tensor.

The most common form of graph is a frozen_model_graph type. These are saved in .pb files, and are serialzed graph defintions with frozen weights. This lets it be used in this example. There are ways to use the saved_model variants, however it is typically better to use the serialized versions. 

These frozen model graph types can be produced from using native tensorflow , or addon frameworks like keras. Also keras .h5 models require a conversion to be useable in native tensorflow, additionally all ops must be tensorflow ops. 

Graphs are required to run any operation. In this repo, the session run function handles most of that for you.

#### Scope
A scope is a way of creating a graph object, operations are added onto a scope with the path the tensor will take. Operations are stacked in the order they will be performed. 

In this case in the ReadTensorFromImageFile:
A root scope is created.
Input/output names are created. This is the same as a beginning and end nodes in frozen ml graphs.
A tensor with the file contents is created. (DT_String)
A file reader op is created. This is a placeholder for the DT string tensor.
A file parsing operation is chosen from ChooseOutputType, adding the op to the root scope.
The selected formatting operations are all added to the root scope in order.(cast to float, expand dims, resize, and div.)
The scope is then converted to a graph and goes through the parsing functions, and then the selected formatting operations in order. This is why the output operations are chained. (dims takes float caster, resize takes dim, etc...)


