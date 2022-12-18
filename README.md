# NYU ECE7123 final project

## Instruction

To make it easier and faster to define the Convolutional Neural Networks we need, we use `nn.Sequential()`. Just like stacking blocks, we can freely add as many layers as we need to `nn.Sequential()` to build the Convolutional Neural Networks.

The following models are equal:

```python
# 1. Traditional method
class Net(nn.Module):
 def __init__(self):
  super(Net, self).__init__()
  self.conv1 = nn.Conv2d(3, 6, 5)
  self.pool = nn.MaxPool2d(2, 2)
  self.conv2 = nn.Conv2d(6, 16, 5)
  self.fc1 = nn.Linear(16 * 5 * 5, 120)
  self.fc2 = nn.Linear(120, 84)
  self.fc3 = nn.Linear(84, 10)
 
 def forward(self, x):
  x = self.pool(F.relu(self.conv1(x)))
  x = self.pool(F.relu(self.conv2(x)))
  x = x.view(-1, 16 * 5 * 5)
  x = F.relu(self.fc1(x))
  x = F.relu(self.fc2(x))
  x = self.fc3(x)
  return x

model = Net()

# 2. nn.Sequential()

model = nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), 
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Linear(120, 84),
        nn.Linear(84, 10)
        )

```

## Group Members

1. Kaiyu Pei
2. Keng-Ming Chang
3. Jincheng Tian
