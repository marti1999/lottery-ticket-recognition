import torch
import torch.nn as nn             #muntar xarxes (capes, activacions, backpropagació de gradients...)
import torch.optim as optim       #escollir optimitzador que recalcularà els pesos
import torch.nn.functional as F   #cridar directament a funcions sense acudir a les classes
import torch.utils.data as data   #muntar dataloaders que generaran els batches de dades

import torchvision                #eines diverses per descarregar bases de dades, transformar dades...

import matplotlib.pyplot as plt   #mostrar, plotejar, displayar dades i imatges

# hiperparàmetres
batch_size = 300                  #per simplificar farem servir mateixa mida per training i test
learning_rate = 0.01              #tasa d'aprenentatge
momentum = 0.9                    #paràmetre pel cas de l'optimitzador SGD (Stochastic gradient descent)
n_epochs = 5                  #vegades que la xarxa veurà totes les dades d'entrenament
criterium = nn.CrossEntropyLoss() #loss function

# dades 
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])   

train_data = torchvision.datasets.MNIST('/files/', train=True,  download=True, transform=trans)
test_data  = torchvision.datasets.MNIST('/files/', train=False, download=True, transform=trans) 

train_loader = data.DataLoader(train_data,batch_size=batch_size, shuffle=True)
test_loader =  data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# exemple de mostra
idx = 13488                       #un qualsevol per veure com són
input_example,target_example = train_data.__getitem__(idx)
print(input_example.size())       #mida de cada mostra/imatge/digit
plt.imshow(input_example[0,:,:], cmap='gray')
plt.title(str(target_example))
plt.show()

# exemple de batch
dataiter = iter(train_loader)
batch_images_example, batch_labels_example = dataiter.next()

plt.imshow(torchvision.utils.make_grid(batch_images_example)[0,:,:], cmap='gray')
plt.title('batch example')
plt.show()
print(batch_labels_example)

# definició de la xarxa
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5)   #def convolutonal layer 
        self.conv2 = nn.Conv2d(10,20,5)  #def convolutonal layer 
        self.fc1 = nn.Linear(20*4*4,50)  #def fully connected layer
        self.fc2 = nn.Linear(50,10)      #def fully connected layer

    def forward(self, x):         #size [100,1,28,28] [batch,channels,height,width]
        x = F.relu(self.conv1(x)) #size [100,1,24,24] padding effect
        x = F.max_pool2d(x,2)     #size [100,1,12,12] downsampling      
        x = F.relu(self.conv2(x)) #size [100,1,8,8]   padding effect
        x = F.max_pool2d(x,2)     #size [100,1,4,4]   downsampling      
        x = torch.flatten(x,1)    #flatten all dimensions except batch
        x = F.relu(self.fc1(x))   #size [100,320] 
        x = self.fc2(x)           #size [100,50]  
        return x                  #size [100,10]

# instanciació de la xarxa i l'optimitzador
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# variables per graficar resultats
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# definició dels bucles d'aprenentatge i test (validació)
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = criterium(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:      #mostra log cada 100 batches
      print('epoch: {:2d} [{:5d}/{} ({:3.0f}%)]\tloss: {:.6f}'.format(
        epoch+1, batch_idx*len(data), len(train_loader.dataset),
        100.0*batch_idx/len(train_loader), loss.item()/batch_size))
      train_losses.append(loss.item()/batch_size)
      train_counter.append(batch_idx*batch_size+epoch*len(train_loader.dataset))

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += criterium(output, target)
      _,pred = torch.max(output.data,1)   
      correct += (pred == target).sum().item() 
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss.item())
  print('\nTest set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:3.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),100.0*correct/len(test_loader.dataset)))    
  return correct

# llencem els experiments
test()
for epoch in range(n_epochs):
  train(epoch)
  correct = test()

# mostrem els resultats
print('\nTest set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:3.0f}%)\n'.format(
    test_losses[-1], correct, len(test_loader.dataset),100.0*correct/len(test_loader.dataset)))     

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')

plt.scatter(test_counter, test_losses, color='red')
plt.legend(['train mean loss', 'test mean loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('mean crossentropy loss')
plt.show