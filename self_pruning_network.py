# SELF-PRUNING NEURAL NETWORK ON CIFAR-10

# Run: python self_pruning_network.py
# Requirements: pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# THE PRUNABLE LINEAR LAYER
class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate_scores for each weight.
    Gates are obtained by applying sigmoid to gate_scores, keeping values in [0,1].
    pruned_weights = weight * sigmoid(gate_scores)
    Gradients flow through both weight and gate_scores automatically via autograd.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features,in_features)*0.01)  # standard weight
        self.bias   = nn.Parameter(torch.zeros(out_features))                   # standard bias 
        # One gate_score per weight (same shape as weight)
        # Initialized near 1 so gates start open (sigmoid(2) ≈ 0.88)
        self.gate_scores = nn.Parameter(torch.zeros(out_features,in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)      # Convert gate_scores → gates in [0, 1] via sigmoid
        pruned_weights = self.weight*gates           # Element-wise multiply weights with gates
        return F.linear(x,pruned_weights,self.bias)  # Standard linear operation  (gradients flow through both)

# NETWORK DEFINITION (Using PrunableLinear instead of nn.Linear)
class SelfPruningNet(nn.Module):
    """
    Simple feed-forward network for CIFAR-10 (32x32x3 → 10 classes).
    All linear layers use PrunableLinear so they can be pruned during training.
    """
    def __init__(self):
        super(SelfPruningNet,self).__init__()
        self.fc1 = PrunableLinear(3*32*32,512)     # input → hidden1
        self.fc2 = PrunableLinear(512,256)         # hidden1 → hidden2
        self.fc3 = PrunableLinear(256,10)          # hidden2 → output

    def forward(self,x):
        x = x.view(x.size(0),-1)         # flatten image to vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                  # raw logits for cross-entropy
        return x

# SPARSITY REGULARIZATION LOSS (helper function)
def sparsity_loss(model):
    """
    L1 penalty = sum of all gate values (sigmoid outputs) across every PrunableLinear layer.  
    Because sigmoid output is always positive,|gate|=gate, so (L1 norm=sum of gates).
    Why L1 drives gates to zero:
      The gradient of sum(gates) w.r.t. gate_scores is always ±sigmoid'(s),
      which is a constant-magnitude push toward zero — unlike L2 which shrinks
      the gradient as the value approaches zero and never fully zeros it out.
      This constant gradient pressure forces many gates to settle exactly at 0.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module,PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total = total + gates.sum()
    return total

# DATA LOADING (CIFAR-10)
def get_dataloaders(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    test_loader  = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=2)
    return train_loader,test_loader

# TRAINING LOOP
def train(model,train_loader,optimizer,lambda_sparse,device):
    """One full epoch of training with combined loss."""
    model.train()
    total_loss = 0.0
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        logits = model(images)                         # Forward pass
        cls_loss = F.cross_entropy(logits,labels)
        sparse_loss = sparsity_loss(model)
        loss = cls_loss + lambda_sparse * sparse_loss  # Total Loss = Classification Loss + λ * Sparsity Loss
        loss.backward()                                # Backward pass—gradients flow to both weight and gate_scores
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

# EVALUATION — accuracy + sparsity level(% of weights whose gate < threshold)
def evaluate(model,test_loader,device):
    """Returns test accuracy(%) and sparsity-level(%)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0*correct/total
    threshold = 1e-2
    pruned = 0
    total_weights = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                pruned += (gates < threshold).sum().item()
                total_weights += gates.numel()
    sparsity = 100.0*pruned/total_weights
    return accuracy,sparsity

# RUN EXPERIMENT FOR ONE λ VALUE
def run_experiment(lambda_sparse,train_loader,test_loader,device,epochs=15):
    """Train a fresh model with the given λ and return results."""
    print(f"\n{'='*55}")
    print(f" Training with λ = {lambda_sparse}")
    print(f"{'='*55}")
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(1,epochs+1):
        avg_loss = train(model,train_loader,optimizer,lambda_sparse,device)
        if epoch%5 == 0:
            acc,spar = evaluate(model,test_loader,device)
            print(f" Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} "
                  f"| Acc: {acc:.1f}% | Sparsity: {spar:.1f}%")
    acc,spar = evaluate(model,test_loader,device)
    print(f"\n Final → Accuracy: {acc:.2f}%  |  Sparsity: {spar:.2f}%")
    return model,acc,spar

# PLOT: GATE VALUE DISTRIBUTION FOR BEST MODEL
def plot_gate_distribution(model,lambda_val):
    """
    Plots a histogram of all final gate values.
    A successful pruning shows a large spike near 0 and a cluster away from 0.
    """
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates.tolist())
    all_gates = np.array(all_gates)
    plt.figure(figsize=(8, 4))
    plt.hist(all_gates, bins=100, color='steelblue', edgecolor='white', alpha=0.85)
    plt.title(f'Gate Value Distribution  (λ = {lambda_val})', fontsize=13)
    plt.xlabel('Gate Value (sigmoid output)')
    plt.ylabel('Count')
    plt.axvline(x=0.01, color='red', linestyle='--', label='Prune threshold (0.01)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gate_distribution.png', dpi=150)
    plt.show()
    print("\nPlot saved as gate_distribution.png")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders(batch_size=256) 
    lambdas = [1e-5, 1e-4, 1e-3]   # low, medium, high
    results = {}
    best_model = None
    best_lambda = None
    best_accuracy = -1
    for lam in lambdas:
        model,acc,spar = run_experiment(lam,train_loader,test_loader,device,epochs=15)
        results[lam] = {'accuracy': acc, 'sparsity': spar}
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_lambda = lam

    print("\n\n" + "="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>14}")
    print("="*55)
    for lam in lambdas:
        acc  = results[lam]['accuracy']
        spar = results[lam]['sparsity']
        print(f"  {lam:<12} {acc:>14.2f}% {spar:>13.2f}%")
    print("="*55)
    plot_gate_distribution(best_model, best_lambda)