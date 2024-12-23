import numpy as np
from typing import Tuple
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

class DataLoader:
    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_samples = data.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.indices = np.arange(self.num_samples)
    
    def get_batch(self, batch_index: int) -> Tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        end = min((batch_index + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start:end]
        
        return self.data[batch_indices], self.labels[batch_indices]   
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for batch_index in range(self.num_batches):
            yield self.get_batch(batch_index)
            
    def reset(self) -> None:
        self.indices = np.arange(self.num_samples)

def generate_data(size: int = 200, num_features: int = 2, 
                  A: np.ndarray = None, B: np.ndarray =  None, task: str = "regression", t: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    assert task in ["classification", "regression"]
    if A is None:
        A = np.random.randn(num_features)
    if B is None:
        B = np.random.randn()
    X = np.random.randn(size, num_features)
    Y = X.dot(A) + B
    
    if task == "classification":
        data, labels = X, (Y > t).astype(int).reshape(-1, 1)
    elif task == "regression":
        data, labels = X, Y.astype(int).reshape(-1, 1)
    return data, labels

def binary_cross_entropy_loss(target: np.ndarray, pred: np.ndarray) -> float:
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    return -np.mean(target * np.log(pred) + (1 - target) * (np.log(1 - pred)))

def mean_square_error_loss(target: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(np.square(target - pred))

class BPNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float, task: str = "regression"):
        assert task in  ["classification", "regression"]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.task = task
        self.lr = lr
        # Kaiming初始化
        # self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        # self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size) * np.sqrt(2. / self.input_size)
        # Xavier初始化
        # self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size) * np.sqrt(2. / (self.hidden_size + self.output_size))
        # self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size) * np.sqrt(2. / (self.hidden_size + self.output_size))
        # 随机初始化
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        # self.bias_hidden = np.random.rand(1, self.hidden_size)
        # self.bias_output = np.random.rand(1, self.output_size)

    def __call__(self, *args, **kwds):
        return self.forward(*args)
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(int)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        if self.task == "classification":
            self.final_output = self.sigmoid(self.output)
        elif self.task == "regression":
            self.final_output = self.output
            
        return self.final_output
    def backward(self, pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            loss = binary_cross_entropy_loss(pred, labels)
            output_error = pred - labels
        elif self.task == "regression":
            loss = mean_square_error_loss(pred, labels)
            output_error = 2 * (pred - labels) / labels.shape[0]
         
        grad_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        grad_bias_output = np.sum(output_error, axis=0, keepdims=True)

        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_output)

        grad_weights_input_hidden = np.dot(self.X.T, hidden_error)
        grad_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        self.weights_hidden_output -= self.lr * grad_weights_hidden_output
        self.bias_output -= self.lr * grad_bias_output
        self.weights_input_hidden -= self.lr * grad_weights_input_hidden
        self.bias_hidden -= self.lr * grad_bias_hidden
        return loss
        
        
if __name__ == "__main__":
    # task = "classification"
    task = "regression"
    loss_fn = binary_cross_entropy_loss if task == "classification" else mean_square_error_loss 
    
    num_features = 5
    train_data_size, test_data_size = 1000, 100
    train_batch_size, test_batch_size = 100, 100
    lr = 1e-4
    
    # Y = AX + B
    A = np.random.randn(num_features)
    B = np.random.randn()
    train_data, train_labels = generate_data(train_data_size, num_features, A, B, task=task)
    train_dataloader = DataLoader(train_data, train_labels, batch_size=train_batch_size, shuffle=True)
    test_data, test_labels = generate_data(test_data_size, num_features, A, B, task=task)
    test_dataloader = DataLoader(test_data, test_labels, batch_size=test_batch_size, shuffle=False)
    
    
    model = BPNN(input_size=num_features, hidden_size=num_features, output_size=1, lr=lr, task=task)
    
    epochs = 1000
    for epoch in range(epochs):
        epoch_loss = 0.0
        for index, (data, labels) in enumerate(train_dataloader):
            pred_labels = model(data)
            loss = model.backward(pred_labels, labels)
            loss = loss_fn(pred_labels, labels)
            epoch_loss += loss

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_data):.4f}")
    
    test_loss = 0.0
    for index, (data, labels) in enumerate(test_dataloader):
        pred_labels = model(data)
        loss = loss_fn(pred_labels, labels)
        test_loss += loss
    print(f"Test Loss: {test_loss / len(test_data):.4f}")
