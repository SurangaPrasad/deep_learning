import torch

# evaluation function
def eval(net, data_loader, loss_function):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    net.eval()
    correct = 0.0
    num_images = 0.0
    for i_batch, (images, labels) in enumerate(data_loader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        outs = net(images)

        loss = loss_function(outs, labels)
#         _, preds = outs.max(1)
        preds = outs.argmax(dim=1)
        correct += preds.eq(labels).sum()
        num_images += len(labels)

    acc = correct / num_images
    return acc , loss

# training function
def train(net, train_loader, valid_loader, loss_function, optimizer, epoches):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    # Create empty results dictionary
    train_val_results = {"train_loss": [],
                         "train_acc": [],
                         "val_loss": [],
                         "val_acc":[]
                        }

    for epoch in range(epoches):
        net.train()
        correct = 0.0 # used to accumulate number of correctly recognized images
        num_images = 0.0 # used to accumulate number of images
        for i_batch, (images, labels) in enumerate(train_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # forward propagation
            outs = net(images)
            # backward propagation
            loss = loss_function(outs, labels)
            loss.backward()
            # calculating the accuracy
            preds = outs.argmax(dim=1)
            correct += preds.eq(labels).sum()

            # update parameters
            optimizer.step()
            optimizer.zero_grad()
            num_images += len(labels)


        acc = correct / num_images
        acc_eval,loss_eval = eval(net, valid_loader, loss_function)

        train_val_results["train_loss"].append(loss)
        train_val_results["train_acc"].append(acc)
        train_val_results["val_loss"].append(loss_eval)
        train_val_results["val_acc"].append(acc_eval)

        if epoch % 10 == 9: 
            print('epoch: %d, lr: %f, accuracy: %f, loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], acc, loss.item(), acc_eval))

    return train_val_results
