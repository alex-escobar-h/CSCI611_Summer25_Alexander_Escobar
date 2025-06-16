# A2: CNN

## Required libraries

- To run the Jupyter notebooks, install the following libraries:
  - `torch`
  - `torchvision`
  - `numpy`
  - `matplotlib`

## Assignment Instructions

### Tasks

- [30%] Part 1: follow image_filter.zip from Modules/CourseResources/JupyterNotebooks, complete all the TODOs in there
- [30%] Part 2: follow build_cnn.zip from Modules/CourseResources/JupyterNotebooks, complete all the TODOs in there
- [40%] Write an report summarizing your implementation and findings. Plot a graphical illustration of your CNN architecture. Include any other visualizations you think helpful for the analysis of your implementation and findings. Report the following but not limited to:
  - Accuracy
  - Time taken for training
  - Number of parameters
  - What helps what does not?

### Submission Requirements

- Create a Github repository that includes the following:
- Name of your repository: CSCI611*Summer25*<Firstname>\_<Lastname>
- Within your repository, create a project or folder: Assignment_2
- Within that project or folder, include the following:
  - A readme file describes how to use your code
  - Source code in Jupyter Notebook.
  - Execution trace of the code in Jupyter Notebook with its generated outputs, graphs and tables etc.
  - A PDF file of the report
  - Any additional data, tools, support material that are used in the project
- Submission (both of the following):
  - In Canvas Assignments, submit the PDF report as File Upload
  - In Canvas Assignments, submit the link to your Github repo/projects as Comments.

## Part One

Other image processing/filtering you can try:

- Other Edge Detector (e.g. Sobel Operator)
  A common 3×3 kernel for edge detection is the **Sobel operator**, which detects edges in a specific direction. Below are the Sobel kernels for detecting vertical and horizontal edges:

**Vertical Edge Detection Kernel:**

> $
K_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$

**Horizontal Edge Detection Kernel:**

> $
K_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$

These kernels are convolved with an image to highlight vertical and horizontal edges, respectively.

- Corner Detection (use the kernels we discussed in slides)
- Scaling (after the blurring, can you pick one pixel out of the following?)
  - 2x2
  - 4x4
- Use other images of your choice
- For a challenge, see if you can put the image through a series of filters:
  - first one that blurs the image (takes an average of pixels),
  - and then one that detects the edges.

## Part 2: Define the CNN Network Architecture

- nn.Conv2d(): for convolution
- nn.MaxPool2d(): for maxpooling (spatial resolution reduction)
- nn.Linear(): for last 1 or 2 layers of fully connected layer before the output layer.
- nn.Dropout(): optional, [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) can be used to avoid overfitting.
- F.relu(): Use ReLU as the activation function for all the hidden layers

```python
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # TODO: Build multiple convolutional layers (sees 32x32x3 image tensor in the first hidden layer)
        # for example, conv1, conv2 and conv3
        pass

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # TODO: Build some linear layers (fully connected)
        # for example, fc1 and fc2
        pass

        # TODO: dropout layer (p=0.25, you can adjust)
        # example self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        # assume we have 2 convolutional layers defined agove
        # and we do a maxpooling after each conv layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # TODO: flatten x at this point to get it ready to feed into the fully connected layer(s)
        # Can use this but need to figure out the actual value for a, b and c
        # x = x.view(-1, a * b * c)

        # optional add dropout layer
        #x = self.dropout(x)

        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # optional add dropout layer
        #x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

# create a complete CNN
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
```

- Compare the difference between **ADAM** and **SGD** optimizer.
