##https://github.com/estija/CMID/blob/main/train.py
def test(model, test_loader, args):
  model.eval()
  test_loss = 0
  correct = 0
  corr = 0
  #print(pl)
  with torch.no_grad():
    for batch in test_loader:
      batch = tuple(t.cuda() for t in batch)
      data = batch[0]
      target = batch[1]
      gl = batch[2]

      data, target, gl = data.cuda(), target.cuda().float(), gl.cuda()
      if args.dataset=='MultiNLI':
          target=target.long()
     
      output = model(data.float())
      if args.dataset=='MultiNLI':
          test_loss += F.cross_entropy(torch.squeeze(output), target, reduction='sum').item()  # sum up batch loss
          pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      corr += pred.eq(gl.float().view_as(pred)).sum().item()
  test_loss /= len(test_loader.dataset)

  print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Correlation: ({:.2f})\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset),
    corr / len(test_loader.dataset)))